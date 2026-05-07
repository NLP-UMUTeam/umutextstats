import pandas as pd

from umutextstats.cli.main import main


def test_cli_analyze_computes_offensive_dictionary(tmp_path, monkeypatch):
    dataset_path = tmp_path / "dataset.csv"
    output_path = tmp_path / "features.csv"
    
    config_path = tmp_path / "config.xml"

    config_path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
    <configuration>
      <directory_folder>es</directory_folder>
      <dimensions>
        <dimension>
          <key>register-offensive-speech-general</key>
          <class>WordPerDictionary</class>
          <dictionary>offensive</dictionary>
        </dimension>
      </dimensions>
    </configuration>
    """,
        encoding="utf-8",
    )

    df = pd.DataFrame(
        {
            "id": [1, 2],
            "tweet": [
                "mierda puta imbécil",
                "hola mundo amable",
            ],
        }
    )

    df.to_csv(dataset_path, index=False)

    monkeypatch.setattr(
        "sys.argv",
        [
            "umutextstats",
            "analyze",
            str(dataset_path),
            "-t",
            "tweet",
            "-c",
            str(config_path),
            "-o",
            str(output_path),
            "--no-cache",
            "--no-stanza",
        ],
    )

    main()

    features = pd.read_csv(output_path)

    assert "register-offensive-speech-general" in features.columns

    value = features.loc[0, "register-offensive-speech-general"]
    
    print (features)

    assert value > 0