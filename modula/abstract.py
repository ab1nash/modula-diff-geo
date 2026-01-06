import jax
import copy

class Module:
    def __init__(self):
        self.children = []

        self.atoms = None           # number of atoms: int
        self.bonds = None           # number of bonds: int
        self.smooth = None          # is this module smooth?: bool
        self.sensitivity = None     # input Lipschitz estimate: float > 0
        self.mass = None            # proportional contribution of module toward feature learning of any supermodule: float >= 0

    def __str__(self):
        string = self.__class__.__name__
        string += f"\n...consists of {self.atoms} atoms and {self.bonds} bonds"
        string += f"\n...{'smooth' if self.smooth else 'non-smooth'}"
        string += f"\n...input sensitivity is {self.sensitivity}"
        string += f"\n...contributes proportion {self.mass} to feature learning of any supermodule"
        return string

    def tare(self, absolute=1.0, relative=None):
        if relative is None:
            self.tare(relative = absolute / self.mass)
        else:
            self.mass *= relative
            for childModule in self.children:
                childModule.tare(relative=relative)

    def jit(self):
        self.forward = jax.jit(self.forward)
        self.project = jax.jit(self.project)
        self.dualize = jax.jit(self.dualize)

    def forward(self, inputData, weightsList):
        # Input and weight list --> output
        raise NotImplementedError

    def initialize(self, key):
        # Return a weight list.
        raise NotImplementedError

    def project(self, weightsList):
        # Return a weight list.
        raise NotImplementedError

    def dualize(self, weightGradsList, targetNorm):
        # Weight gradient list and number --> normalized weight gradient list
        raise NotImplementedError

    def __matmul__(self, otherModule):
        if isinstance(otherModule, tuple):
            otherModule = TupleModule(otherModule)
        return CompositeModule(self, otherModule)

    def __add__(self, otherModule):
        return Add() @ TupleModule((self, otherModule))

    def __mul__(self, scalar):
        assert scalar != 0, "cannot multiply a module by zero"
        return self @ Mul(scalar)

    def __rmul__(self, scalar):
        return Mul(scalar) @ self

    def __pow__(self, exponent):
        assert exponent >= 0 and exponent % 1 == 0, "nonnegative integer powers only"
        return (
            copy.deepcopy(self) @ (self ** (exponent - 1))
            if exponent > 0
            else Identity()
        )

    def __call__(self, inputData, weightsList):
        return self.forward(inputData, weightsList)


class Atom(Module):
    def __init__(self):
        super().__init__()
        self.atoms = 1
        self.bonds = 0

class Bond(Module):
    def __init__(self):
        super().__init__()
        self.atoms = 0
        self.bonds = 1
        self.mass = 0

    def initialize(self, key):
        return []

    def project(self, weightsList):
        return []

    def dualize(self, weightGradsList, targetNorm=1.0):
        return []


class CompositeModule(Module):

    def __init__(self, outerModule, innerModule):
        super().__init__()
        self.children = (innerModule, outerModule)

        self.atoms = innerModule.atoms + outerModule.atoms
        self.bonds = innerModule.bonds + outerModule.bonds
        self.smooth = innerModule.smooth and outerModule.smooth
        self.mass = innerModule.mass + outerModule.mass
        self.sensitivity = innerModule.sensitivity * outerModule.sensitivity

    def forward(self, inputData, weightsList):
        innerModule, outerModule = self.children
        innerWeightsList = weightsList[: innerModule.atoms]
        outerWeightsList = weightsList[innerModule.atoms :]
        intermediateOutput = innerModule.forward(inputData, innerWeightsList)
        finalOutput = outerModule.forward(intermediateOutput, outerWeightsList)
        return finalOutput

    def initialize(self, key):
        innerModule, outerModule = self.children
        key, subkey = jax.random.split(key)
        return innerModule.initialize(key) + outerModule.initialize(subkey)

    def project(self, weightsList):
        innerModule, outerModule = self.children
        innerWeightsList = weightsList[: innerModule.atoms]
        outerWeightsList = weightsList[innerModule.atoms :]
        return innerModule.project(innerWeightsList) + outerModule.project(
            outerWeightsList
        )

    def dualize(self, weightGradsList, targetNorm=1.0):
        if self.mass > 0:
            innerModule, outerModule = self.children
            innerWeightGradsList, outerWeightGradsList = (
                weightGradsList[: innerModule.atoms],
                weightGradsList[innerModule.atoms :],
            )
            innerDualWeightsList = innerModule.dualize(
                innerWeightGradsList,
                targetNorm=targetNorm
                * innerModule.mass
                / self.mass
                / outerModule.sensitivity,
            )
            outerDualWeightsList = outerModule.dualize(
                outerWeightGradsList,
                targetNorm=targetNorm * outerModule.mass / self.mass,
            )
            dualWeightsList = innerDualWeightsList + outerDualWeightsList
        else:
            dualWeightsList = [
                0 * gradWeightMatrix for gradWeightMatrix in weightGradsList
            ]
        return dualWeightsList


class TupleModule(Module):

    def __init__(self, pythonTupleOfModules):
        super().__init__()
        self.children = pythonTupleOfModules
        self.atoms = sum(childModule.atoms for childModule in self.children)
        self.bonds = sum(childModule.bonds for childModule in self.children)
        self.smooth = all(childModule.smooth for childModule in self.children)
        self.mass = sum(childModule.mass for childModule in self.children)
        self.sensitivity = sum(childModule.sensitivity for childModule in self.children)

    def forward(self, inputData, weightsList):
        outputList = []
        for childModule in self.children:
            childOutput = childModule.forward(
                inputData, weightsList[: childModule.atoms]
            )
            outputList.append(childOutput)
            weightsList = weightsList[childModule.atoms :]
        return outputList

    def initialize(self, key):
        weightsList = []
        for childModule in self.children:
            key, subkey = jax.random.split(key)
            weightsList += childModule.initialize(subkey)
        return weightsList

    def project(self, weightsList):
        projectedWeightsList = []
        for childModule in self.children:
            childProjectedWeightsList = childModule.project(
                weightsList[: childModule.atoms]
            )
            projectedWeightsList += childProjectedWeightsList
            weightsList = weightsList[childModule.atoms :]
        return projectedWeightsList

    def dualize(self, weightGradsList, targetNorm=1.0):
        if self.mass > 0:
            dualWeightsList = []
            for childModule in self.children:
                childWeightGradsList = weightGradsList[: childModule.atoms]
                childDualWeightsList = childModule.dualize(
                    childWeightGradsList,
                    targetNorm=targetNorm * childModule.mass / self.mass,
                )
                dualWeightsList += childDualWeightsList
                weightGradsList = weightGradsList[childModule.atoms :]
        else:
            dualWeightsList = [
                0 * gradWeightMatrix for gradWeightMatrix in weightGradsList
            ]
        return dualWeightsList


class Identity(Bond):
    def __init__(self):
        super().__init__()
        self.smooth = True
        self.sensitivity = 1

    def forward(self, inputData, weightsList):
        return inputData


class Add(Bond):
    def __init__(self):
        super().__init__()
        self.smooth = True
        self.sensitivity = 1

    def forward(self, inputData, weightsList):
        return sum(inputData)


class Mul(Bond):
    def __init__(self, scalar):
        super().__init__()
        self.smooth = True
        self.sensitivity = scalar

    def forward(self, inputData, weightsList):
        return inputData * self.sensitivity
