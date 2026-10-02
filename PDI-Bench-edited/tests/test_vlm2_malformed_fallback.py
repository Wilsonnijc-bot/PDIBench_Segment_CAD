import io
import json
import os
import unittest
from unittest.mock import patch

from PIL import Image
import numpy as np

from persistent_masking.interface.config import role_config
from persistent_masking.pipeline import validate_vlm2_call_images
from persistent_masking.vlm_client import VLMClient
from persistent_masking.vlm2_sam_prompting import ask_point_response


class FakeRoute:
    def __init__(self, response, model="gemini-3.8-flash"):
        self.response = response
        self.model = model
        self.calls = 0

    def ask(self, images, prompt, *, system_prompt):
        self.calls += 1
        return {"answer": self.response, "model": self.model}


class VLM2FallbackTests(unittest.TestCase):
    def test_same_endpoint_and_key_with_high_reasoning(self):
        primary = role_config("vlm2")
        fallback = role_config("vlm2_malformed_fallback")
        self.assertEqual(fallback["model"], "gpt-6-luna")
        self.assertEqual(fallback["reasoning_effort"], "high")
        self.assertEqual(fallback["api_base"], primary["api_base"])
        self.assertEqual(fallback["api_key_env"], primary["api_key_env"])

    def test_luna_request_uses_high_reasoning_chat_payload(self):
        sent = []

        def respond(request, timeout):
            sent.append(request)
            return io.BytesIO(json.dumps({"choices": [{"message": {
                "content": '{"negative_points": [[10, 20], [30, 40]]}'}}]}).encode())

        with patch.dict(os.environ, {"VLM2_API_KEY": "dummy-test-key"}), \
             patch("persistent_masking.vlm_client.urllib.request.urlopen", respond):
            VLMClient("test", role_config("vlm2_malformed_fallback")).ask(
                [Image.new("RGB", (2, 2))], "prompt")
        request = sent[0]
        payload = json.loads(request.data)
        self.assertEqual(request.full_url, "https://api.302ai.cn/v1/chat/completions")
        self.assertEqual(payload["model"], "gpt-6-luna")
        self.assertEqual(payload["reasoning_effort"], "high")
        self.assertEqual(payload["max_completion_tokens"], 4096)
        self.assertNotIn("temperature", payload)

    def test_empty_visible_response_records_provider_diagnostic(self):
        response = {"model": "gemini-3.8-flash", "choices": [{
            "finish_reason": "length", "message": {"content": "", "reasoning_content": "hidden"},
        }], "usage": {"completion_tokens": 4096,
                      "completion_tokens_details": {"reasoning_tokens": 4096}}}

        def respond(request, timeout):
            return io.BytesIO(json.dumps(response).encode())

        client = VLMClient("test", role_config("vlm2"))
        with patch.dict(os.environ, {"VLM2_API_KEY": "dummy-test-key"}), \
             patch("persistent_masking.vlm_client.urllib.request.urlopen", respond), \
             self.assertRaisesRegex(ValueError, "missing text"):
            client.ask([Image.new("RGB", (2, 2))], "prompt")
        diagnostic = client.records[-1]["response_diagnostic"]
        self.assertEqual(diagnostic["finish_reason"], "length")
        self.assertEqual(diagnostic["usage"]["completion_tokens"], 4096)
        self.assertEqual(diagnostic["content_type"], "str")
        self.assertNotIn("hidden", json.dumps(diagnostic))

    def test_malformed_response_calls_luna_once_and_records_both(self):
        primary = FakeRoute('{"positive_points": [[1, 2],')
        instances = []

        def make_fallback(role, config):
            route = FakeRoute('{"positive_points": [[10, 20], [30, 40], [50, 60]]}',
                              model="gpt-6-luna")
            instances.append((role, config, route))
            return route

        calls = []
        with patch("persistent_masking.vlm2_sam_prompting.VLMClient", make_fallback):
            result = ask_point_response(primary, [], "prompt", "positive_points",
                                        "positive_points", calls)
        self.assertEqual(primary.calls, 1)
        self.assertEqual(instances[0][2].calls, 1)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1]["role"], "positive_points_luna_fallback")
        self.assertEqual(result["model"], "gpt-6-luna")

    def test_valid_points_and_explicit_null_do_not_use_luna(self):
        for response in ('{"negative_points": [[10, 20], [30, 40]]}',
                         '{"positive_points": null}'):
            key = "negative_points" if "negative_points" in response else "positive_points"
            primary = FakeRoute(response)
            with patch("persistent_masking.vlm2_sam_prompting.VLMClient") as fallback:
                result = ask_point_response(primary, [], "prompt", key, key, [])
            fallback.assert_not_called()
            self.assertEqual(result["answer"], response)

    def test_missing_primary_text_calls_luna_once(self):
        class MissingText:
            model = "gemini-3.8-flash"

            def ask(self, images, prompt, *, system_prompt):
                raise ValueError("VLM response is missing text")

        calls = []
        with patch("persistent_masking.vlm2_sam_prompting.VLMClient") as factory:
            factory.return_value.ask.return_value = {
                "answer": '{"negative_points": [[10, 20], [30, 40]]}',
                "model": "gpt-6-luna",
            }
            ask_point_response(MissingText(), [], "prompt", "negative_points",
                               "negative_points", calls)
        factory.return_value.ask.assert_called_once()
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["error"], "VLM response is missing text")

    def test_one_alternate_attempt_is_shared_across_both_point_requests(self):
        budget = {"remaining": 1}
        calls = []
        primary = FakeRoute('{"positive_points": null}')
        with patch("persistent_masking.vlm2_sam_prompting.VLMClient") as factory:
            factory.return_value.ask.return_value = {
                "answer": '{"positive_points": [[10, 20], [30, 40], [50, 60]]}',
                "model": "gpt-6-luna",
            }
            ask_point_response(primary, [], "prompt", "positive_points",
                               "positive_points", calls, fallback_budget=budget,
                               fallback_on_null=True)
            self.assertEqual(budget["remaining"], 0)
            with self.assertRaisesRegex(ValueError, "one alternate attempt"):
                ask_point_response(FakeRoute('{"negative_points": ['), [], "prompt",
                                   "negative_points", "negative_points", calls,
                                   fallback_budget=budget)
            factory.assert_called_once()
        self.assertEqual(len(calls), 3)

    def test_validation_accepts_one_same_frame_alternate(self):
        frame = np.zeros((4, 5, 3), dtype=np.uint8)
        buffer = io.BytesIO()
        Image.fromarray(frame).save(buffer, format="PNG")
        import hashlib
        digest = hashlib.sha256(buffer.getvalue()).hexdigest()
        hashes = [digest, "a", "b", "c"]
        seed = {"earliest_deformed_frame": 3, "calls": [
            {"role": "positive_points", "image_sha256": hashes},
            {"role": "positive_points_luna_fallback", "image_sha256": hashes},
            {"role": "negative_points", "image_sha256": hashes},
        ]}
        with patch("persistent_masking.pipeline.read_original_frame", return_value=frame):
            validate_vlm2_call_images(seed, [{"frame": 3, "state": "deformed"}],
                                      "video.mp4", digest, hashes[1:])


if __name__ == "__main__":
    unittest.main()
