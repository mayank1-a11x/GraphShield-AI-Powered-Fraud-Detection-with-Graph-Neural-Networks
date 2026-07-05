import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from src import config
from src import data_preprocessing


class DataPreprocessingTests(unittest.TestCase):
    def test_load_raw_data_supports_workbook_input(self):
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            workbook_path = tmp_path / "fraud_synthetic_dataset.xlsx"

            df = pd.DataFrame(
                [
                    {
                        "TransactionID": 1,
                        "isFraud": 0,
                        "TransactionAmt": 50.0,
                        "DeviceType": "mobile",
                        "P_emaildomain": "gmail.com",
                    },
                    {
                        "TransactionID": 2,
                        "isFraud": 1,
                        "TransactionAmt": 200.0,
                        "DeviceType": "desktop",
                        "P_emaildomain": "outlook.com",
                    },
                ]
            )
            df.to_excel(workbook_path, index=False)

            with patch.object(config, "TRANSACTION_CSV", str(tmp_path / "train_transaction.csv")), patch.object(
                config, "IDENTITY_CSV", str(tmp_path / "train_identity.csv")
            ), patch.object(config, "WORKBOOK_PATH", str(workbook_path)):
                result = data_preprocessing.load_raw_data()

            self.assertEqual(result.shape[0], 2)
            self.assertIn("isFraud", result.columns)
            self.assertIn("TransactionID", result.columns)
            self.assertEqual(result["isFraud"].sum(), 1)


if __name__ == "__main__":
    unittest.main()
