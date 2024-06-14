from typing import List
from rich import print

from perturbench.metrics import PerturbationMetric
from perturbench.models import PerturbationModel
from perturbench.dataset import PerturbationDataset
from perturbench.scenarios import PerturbationScenario


class PerturbationBenchmark:
    """Responsible for comparing performance across different scenarios / models"""

    def __init__(self):
        self.models = []
        self.scenarios = []
        self.metrics = []
        self.datasets = []
        pass

    def register_dataset(self, dataset: PerturbationDataset):
        print(f"Registering {dataset.get_name()}")
        self.datasets = self.datasets + [dataset]

    def register_scenario(self, scenario: PerturbationScenario):
        self.scenarios.append(scenario)

    def register_model(self, model: PerturbationModel):
        self.models.append(model)

    def register_metric(self, metric: PerturbationMetric):
        self.metrics.append(metric)

    def registered(self):

        print(f"Registered {len(self.datasets)} datasets:")
        for dataset in self.datasets:
            print(f"[bold red] - {dataset.get_name()}: [/bold red]")
            print(f"    |- [bold]description:[/bold] {dataset.get_description()}")
            print(f"    |- [bold] perturbations:[/bold] {dataset.get_perturbations().unique().tolist()}")

        print(f"Registered {len(self.scenarios)} scenarios:")
        for scenario in self.scenarios:
            print(f"[bold red] - {scenario.get_name()}: [/bold red]")
            print(f"    |- [bold]description:[/bold] {scenario.get_description()}")

        print(f"Registered {len(self.models)} models:")
        for model in self.models:
            print(f"[bold red] - {model.get_name()}: [/bold red]")
            print(f"    |- [bold]description:[/bold] {model.get_description()}")

        print(f"Registered {len(self.metrics)} metrics:")
        for metric in self.metrics:
            print(f"[bold red] - {metric.get_name()}: [/bold red]")
            print(f"    |- [bold]description:[/bold] {metric.get_description()}")

    # def run(self):
    #    for dataset in self.datasets:
    #        print(f"Starting run for dataset {dataset.get_name()}")
    #
    #         dataset_anndata = dataset.get_anndata()
    #         control_dataset_anndata = dataset_anndata[dataset_anndata.obs["perturbation"] is None, :]
    #         control_dataset = PerturbationDataset(
    #             anndata=control_dataset_anndata,
    #             perturbation_field="perturbation",
    #             covariate_fields=dataset.covariates().columns.tolist(),
    #             name=dataset.get_name(),
    #             description=dataset.get_description()
    #         )
    #
    #         for scenario in self.scenarios:
    #             print(f"Starting run for scenario {scenario.get_name()}")
    #
    #             test_dataset, train_dataset = scenario.split(dataset)
    #
    #             for model in self.models:
    #                 print(f"Starting run for model {model.get_name()}")
    #
    #                 model.train(train_dataset)
    #
    #                 prediction = model.predict(
    #                     data=test_dataset,
    #                     perturbation=[x for x in dataset.get_perturbations().unique().tolist() if x is not None]
    #                 )
    #
    #                 for metric in self.metrics:
    #                     print(f"Starting run for metric {metric.get_name()}")
    #                     calculated_metric = metric.compute()
