import jax
import jax.numpy as jnp

from modula.abstract import Atom


def orthogonalize(inputMatrix):
    # For a given matrix, reduce all singular values to 1.
    # UΣV^T ~> UV^T
    # six step Newton-Schulz by @YouJiacheng
    # coefficients from: https://twitter.com/YouJiacheng/status/1893704552689303901
    # found by optimization: https://gist.github.com/YouJiacheng/393c90cbdc23b09d5688815ba382288b/5bff1f7781cf7d062a155eecd2f13075756482ae
    # the idea of stability loss was from @leloykun

    newtonSchulzCoeffs = [
        (3955 / 1024, -8306 / 1024, 5008 / 1024),
        (3735 / 1024, -6681 / 1024, 3463 / 1024),
        (3799 / 1024, -6499 / 1024, 3211 / 1024),
        (4019 / 1024, -6385 / 1024, 2906 / 1024),
        (2677 / 1024, -3029 / 1024, 1162 / 1024),
        (2172 / 1024, -1833 / 1024, 682 / 1024),
    ]

    transpose = inputMatrix.shape[1] > inputMatrix.shape[0]
    if transpose:
        inputMatrix = inputMatrix.T
    inputMatrix = inputMatrix / jnp.linalg.norm(inputMatrix)
    for coeffA, coeffB, coeffC in newtonSchulzCoeffs:
        inputGramMatrix = inputMatrix.T @ inputMatrix
        identityMatrix = jnp.eye(inputGramMatrix.shape[0])
        inputMatrix = inputMatrix @ (
            coeffA * identityMatrix
            + coeffB * inputGramMatrix
            + coeffC * inputGramMatrix @ inputGramMatrix
        )
    if transpose:
        inputMatrix = inputMatrix.T
    return inputMatrix


class Linear(Atom):
    def __init__(self, fanout, fanin):
        super().__init__()
        self.fanin  = fanin
        self.fanout = fanout
        self.smooth = True
        self.mass = 1
        self.sensitivity = 1

    def forward(self, inputData, weightsList):
        # inputData shape is [..., fanin]
        weightMatrix = weightsList[0]  # shape is [fanout, fanin]
        return jnp.einsum("...ij,...j->...i", weightMatrix, inputData)

    def initialize(self, key):
        weightMatrix = jax.random.normal(key, shape=(self.fanout, self.fanin))
        weightMatrix = orthogonalize(weightMatrix) * jnp.sqrt(self.fanout / self.fanin)
        return [weightMatrix]

    def project(self, weightsList):
        weightMatrix = weightsList[0]
        weightMatrix = orthogonalize(weightMatrix) * jnp.sqrt(self.fanout / self.fanin)
        return [weightMatrix]

    def dualize(self, weightGradsList, targetNorm=1.0):
        weightGradMatrix = weightGradsList[0]
        dualWeightMatrix = (
            orthogonalize(weightGradMatrix)
            * jnp.sqrt(self.fanout / self.fanin)
            * targetNorm
        )
        return [dualWeightMatrix]


class Embed(Atom):

    def __init__(self, dEmbed, numEmbed):
        super().__init__()
        self.numEmbed = numEmbed
        self.dEmbed = dEmbed
        self.smooth = True
        self.mass = 1
        self.sensitivity = 1

    def forward(self, inputData, weightsList):
        weightMatrix = weightsList[0]  # shape [numEmbed, dEmbed]
        return weightMatrix[inputData]

    def initialize(self, key):
        weightMatrix = jax.random.normal(key, shape=(self.numEmbed, self.dEmbed))
        weightMatrix = (
            weightMatrix
            / jnp.linalg.norm(weightMatrix, axis=1, keepdims=True)
            * jnp.sqrt(self.dEmbed)
        )
        return [weightMatrix]

    def project(self, weightsList):
        weightMatrix = weightsList[0]
        weightMatrix = (
            weightMatrix
            / jnp.linalg.norm(weightMatrix, axis=1, keepdims=True)
            * jnp.sqrt(self.dEmbed)
        )
        return [weightMatrix]

    def dualize(self, weightGradsList, targetNorm=1.0):
        weightGradMatrix = weightGradsList[0]
        dualWeightMatrix = (
            weightGradMatrix
            / jnp.linalg.norm(weightGradMatrix, axis=1, keepdims=True)
            * jnp.sqrt(self.dEmbed)
            * targetNorm
        )
        dualWeightMatrix = jnp.nan_to_num(dualWeightMatrix)
        return [dualWeightMatrix]


if __name__ == "__main__":

    key = jax.random.PRNGKey(0)

    # sample a random numRows x numCols matrix
    numRows, numCols = 50, 100
    randomMatrix = jax.random.normal(key, shape=(numRows, numCols))
    orthogonalizedMatrix = orthogonalize(randomMatrix)

    # compute SVD of randomMatrix and orthogonalizedMatrix
    U, S, Vh = jnp.linalg.svd(randomMatrix, full_matrices=False)
    orthogonalizedSingularValues = jnp.linalg.svd(
        orthogonalizedMatrix, compute_uv=False
    )

    # print singular values
    print(
        f"min singular value of orthogonalizedMatrix: {jnp.min(orthogonalizedSingularValues)}"
    )
    print(
        f"max singular value of orthogonalizedMatrix: {jnp.max(orthogonalizedSingularValues)}"
    )

    print(f"min singular value of randomMatrix: {jnp.min(S)}")
    print(f"max singular value of randomMatrix: {jnp.max(S)}")

    # check that randomMatrix is close to its SVD
    errorRandomMatrix = jnp.linalg.norm(
        randomMatrix - U @ jnp.diag(S) @ Vh
    ) / jnp.linalg.norm(randomMatrix)
    errorOrthogonalizedMatrix = jnp.linalg.norm(
        orthogonalizedMatrix - U @ Vh
    ) / jnp.linalg.norm(U @ Vh)
    print(f"relative error in randomMatrix's SVD: {errorRandomMatrix}")
    print(f"relative error in orthogonalizedMatrix: {errorOrthogonalizedMatrix}")
