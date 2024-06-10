
newcharge = ""

charge = "CHARGE=-1"

print("charge ---->: ", charge)

if charge.startswith("CHARGE=-"):
    splited_charge = charge.split('-')
    newcharge = splited_charge[0]+splited_charge[1]+'-'
    print(" The charge starts with a negative sign in front charge= ", charge)
else:
    print(" The charge starts with a positive sign in front charge= ", charge)
    newcharge = charge+'+'
# [end-if] statement

print(" The rearranged charge is now given by newcharge = ", newcharge)

IONMODE="positive"
print(" The ionmode before capitalisation = ", IONMODE)
print(" The ionmode before capitalisation = ", IONMODE.capitalize())

IONMODE="POSITIVE"
print(" The ionmode before capitalisation = ", IONMODE)
print(" The ionmode before capitalisation = ", IONMODE.capitalize())

IONMODE="pOsITive"
print(" The ionmode before capitalisation = ", IONMODE)
print(" The ionmode before capitalisation = ", IONMODE.capitalize())


