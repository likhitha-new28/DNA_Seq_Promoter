from dna_classifier.cli import main


def test_demo_data_command_creates_csv(tmp_path, monkeypatch):
    output = tmp_path / "demo.csv"
    monkeypatch.setattr(
        "sys.argv",
        ["dna-classifier", "demo-data", "--output", str(output), "--samples-per-class", "2"],
    )
    main()
    lines = output.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "sequence,label"
    assert len(lines) == 7
