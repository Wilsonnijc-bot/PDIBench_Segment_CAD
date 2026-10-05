"""CLI shortcut; the coordinator and its imports live in infrastructure."""
if __name__ == '__main__':
    from infrastructure.deformation_detect.__main__ import main
    raise SystemExit(main())
