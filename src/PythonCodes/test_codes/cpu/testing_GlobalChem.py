import global_chem

from global_chem_extensions import GlobalChemExtensions

gc = GlobalChem()
smiles_list = gc.get_all_smiles()

sucesses, failures = GlobalChemExtensions.verify_smiles(
    smiles_list,
    rdkit=True,
    partial_smiles=False,
    return_failures=True,
    pysmiles=False,
    molvs=False
)
print (failures)

