"""Out-of-band digest binding for the immutable v2r4 gate contract.

This is the sole source file excluded from the contract's project-tree
manifest.  The contract is generated only after every other gate source is
frozen; these two values are then filled with the canonical self-hash and
byte hash of that external contract file.
"""

SUPERSEDED_V2R4_CONTRACT_SHA256 = (
    "3127bdd6dfca62b34813e3fe938300d5d44c8d7ac253bf4a65836f4b2fc1ffd3"
)
SUPERSEDED_V2R4_CONTRACT_FILE_SHA256 = (
    "9bc355fb7ac89dda15cbf4d0c1a4767a3ac5e314e6c800c398f3e9062de02f29"
)

EXPECTED_CONTRACT_SHA256 = (
    "3e201cfd9094815cf72a63058d4225b334e3e5cd77fc0d79fbe6379d48778c9d"
)
EXPECTED_CONTRACT_FILE_SHA256 = (
    "04945f691aa55e3aaa860851b33b49aba9eea789533e5f86f3e0d2345bbc1c38"
)
