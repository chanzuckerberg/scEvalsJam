import numpy as np

from sams_vae.data.norman.data_module import (
    NormanDataEfficiencyDataModule,
    NormanOODCombinationDataModule,
)


class TestNormanOODCombinationDataModule:
    def test_splits(self):
        # initialize dummy data module
        data_module = NormanOODCombinationDataModule(
            frac_combinations_train=0.2,
        )

        # generate split labels for various frac_train and split_seed values
        obs = data_module.adata.obs.copy()
        frac_train_list = np.linspace(0, 1, 6)
        split_seeds = [0, 1]
        for i, frac_train in enumerate(frac_train_list):
            for j, split_seed in enumerate(split_seeds):
                split_labels = data_module._get_split_labels(
                    obs=obs,
                    frac_combinations_train=frac_train,
                    frac_combinations_test=0.2,
                    split_seed=split_seed,
                )

                obs[f"split{i}_{j}"] = split_labels

        # check that splits accumulate samples
        for split_seed in split_seeds:
            for i in range(len(frac_train_list) - 1):
                assert np.all(
                    obs[obs[f"split{i}_{split_seed}"] == "train"][
                        f"split{i+1}_{split_seed}"
                    ]
                    == "train"
                )
                assert np.all(
                    obs[obs[f"split{i}_{split_seed}"] == "val"][
                        f"split{i+1}_{split_seed}"
                    ]
                    == "val"
                )
                assert np.all(
                    obs[obs[f"split{i}_{split_seed}"] == "test"][
                        f"split{i+1}_{split_seed}"
                    ]
                    == "test"
                )

        # check that fraction of cells follows expected distribution
        combos = obs[obs["num_guides"] == 2]["guide_identity"].astype(str).unique()
        for split_seed in split_seeds:
            for i, frac_train in enumerate(frac_train_list):
                obs_combo = obs[obs["num_guides"] == 2]

                # fraction of combinations included in training
                train_combos = (
                    obs_combo[
                        obs_combo[f"split{i}_{split_seed}"].isin(["train", "val"])
                    ]["guide_identity"]
                    .astype(str)
                    .unique()
                )
                test_combos = (
                    obs_combo[obs_combo[f"split{i}_{split_seed}"] == "test"][
                        "guide_identity"
                    ]
                    .astype(str)
                    .unique()
                )

                assert np.isclose(
                    len(train_combos) / len(combos), frac_train, atol=0.01
                )
                assert np.isclose(len(test_combos) / len(combos), 0.2, atol=0.01)

                if frac_train < 1 - 0.2:
                    assert len(set(train_combos).intersection(set(test_combos))) == 0

        # check that different split seeds yield different splits
        i = len(frac_train_list) - 1
        assert np.any(obs[f"split{i}_0"].to_numpy() != obs[f"split{i}_1"].to_numpy())


class TestNormanDataEfficiencyDataModule:
    def test_splits(self):
        # initialize dummy data module
        data_module = NormanDataEfficiencyDataModule(
            frac_combination_cells_train=0.5,
        )

        # generate split labels for various frac_train and split_seed values
        obs = data_module.adata.obs.copy()
        frac_train_list = np.linspace(0, 1, 6)
        split_seeds = [0, 1]
        for i, frac_train in enumerate(frac_train_list):
            for j, split_seed in enumerate(split_seeds):
                split_labels = data_module._get_split_labels(
                    obs=obs,
                    frac_combination_cells_train=frac_train,
                    split_seed=split_seed,
                )

                obs[f"split{i}_{j}"] = split_labels

        # check that splits accumulate samples
        for split_seed in split_seeds:
            for i in range(len(frac_train_list) - 1):
                assert np.all(
                    obs[obs[f"split{i}_{split_seed}"] == "train"][
                        f"split{i+1}_{split_seed}"
                    ]
                    == "train"
                )
                assert np.all(
                    obs[obs[f"split{i}_{split_seed}"] == "val"][
                        f"split{i+1}_{split_seed}"
                    ]
                    == "val"
                )
                assert np.all(
                    obs[obs[f"split{i}_{split_seed}"] == "test"][
                        f"split{i+1}_{split_seed}"
                    ]
                    == "test"
                )

        # check that fraction of cells follows expected distribution
        for split_seed in split_seeds:
            for i, frac_train in enumerate(frac_train_list):
                obs_combo = obs[obs["num_guides"] == 2]
                assert np.isclose(
                    (obs_combo[f"split{i}_{split_seed}"] == "train").mean(),
                    frac_train * 0.8 * 0.8,
                    atol=0.01,
                )

                assert np.isclose(
                    (obs_combo[f"split{i}_{split_seed}"] == "val").mean(),
                    0.8 * 0.2,
                    atol=0.01,
                )

                assert np.isclose(
                    (obs_combo[f"split{i}_{split_seed}"] == "test").mean(),
                    0.2,
                    atol=0.01,
                )

        # check that different split seeds yield different splits
        i = len(frac_train_list) - 1
        assert np.any(obs[f"split{i}_0"].to_numpy() != obs[f"split{i}_1"].to_numpy())
