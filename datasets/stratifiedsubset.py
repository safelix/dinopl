from torch.utils.data import Subset
from torch.utils.data.dataset import Dataset

from typing import Dict

class StratifiedSubset(Subset):
    _repr_indent = 4
    def __init__(self, dataset: Dataset, n_samples:int) -> None:

        # count available samples per dataset
        available: Dict[int, int] = {}
        for (_, c) in dataset:
            available[c] = available[c] + 1 if c in available else 0
            if all(size >= n_samples for size in available.keys()):
                break   # early stop if one class could provide all samples

        # stratify stepwise according to the class with the least available samples
        remaining = n_samples  
        stratified: Dict[int, int] = {c:0 for c in available}
        while(min(available.values()) * len(available) <= remaining):
            
            # get class with the fewest samples and stratify its amount among the classes (with more available samples)
            currclass, curramount = min(available.items(), key=(lambda kv: kv[1]))
            for c in sorted(available.keys()):
                stratified[c] += curramount
                available[c] -= curramount

            remaining = n_samples - sum(stratified.values()) # update remaining
            del available[currclass] # delete class from available list



        # now, all classes have more samples available than remaining // len(available)
        assert(min(available.values()) > remaining // len(available))

        # stratify the remaining evenly among those classes which have available samples
        for c in sorted(available.keys()):
            stratified[c] += remaining // len(available)
            available[c] -= remaining // len(available)
        remaining = n_samples - sum(stratified.values()) # update remaining

        # stratify the remaining one at at time according to increasing class index
        for c in sorted(available.keys())[:remaining]:
            stratified[c] += 1
            available[c] -= 1



        # now, all the total stratified quota match the number of samples
        assert(n_samples == sum(stratified.values()))

        # select the indices according tho the stratified quotas
        indices = []
        for idx, (_, c) in enumerate(dataset):
            # add index of sample of the stratified quota allows it
            if stratified[c] > 0:
                indices.append(idx)
                stratified[c] -= 1  # adjust quota

            if len(indices) == n_samples:
                break 


        super().__init__(dataset, indices)

    
    def __repr__(self) -> str:
        head = self.__class__.__name__
        body = [' '*self._repr_indent + line for line in str(self.dataset).splitlines()]
        body += [f'Number of stratified datapoints: {self.__len__()}']

        lines = [head] + [' '*self._repr_indent + line for line in body]
        return "\n".join(lines)