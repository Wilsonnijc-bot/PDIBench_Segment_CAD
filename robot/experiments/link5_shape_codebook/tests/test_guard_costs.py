from robot.experiments.link5_shape_codebook.guard_costs import estimate, usage_tokens


def test_302_zero_aliases_and_reasoning_are_not_double_counted():
    call=dict(model='gemini-3.8-flash',usage=dict(prompt_tokens=1000,completion_tokens=2000,
        input_tokens=0,output_tokens=0,completion_tokens_details=dict(reasoning_tokens=1990)))
    assert usage_tokens(call)['output_tokens']==2000
    assert abs(estimate(call)-.00825)<1e-12


def test_native_usage_and_cached_discount():
    call=dict(requested_model='gemini-3.8-flash',usage=dict(billing_usage=dict(gemini_usage_metadata=dict(
        promptTokenCount=1000,candidatesTokenCount=20,thoughtsTokenCount=1980,cachedContentTokenCount=400))))
    assert usage_tokens(call)['output_tokens']==2000
    assert abs(estimate(call)-.00798)<1e-12


def test_missing_text_can_still_be_billable_and_missing_usage_is_unknown():
    call=dict(model='gpt-6-luna',response_error='No text',response_diagnostic=dict(
        usage=dict(input_tokens=1000,output_tokens=4000)))
    assert abs(estimate(call)-.0021)<1e-12
    assert estimate(dict(model='gpt-6-luna',response_error='Timed out')) is None
