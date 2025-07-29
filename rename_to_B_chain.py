from renu import pdb


# # # This line of code rewrites the md 1bxw protein as A chain
# foo = pdb("inputs/1bxw_default_dppc-tail-contacts.pdb")

# for chain in foo.lines.keys():
#     for resi in foo.lines[chain]:
#         for atom in foo.lines[chain][resi]:
#             print(atom.set_chain("A"))
# foo.write_pdb(path='1bxw_chainA.pdb',new_lines=True)



foo = pdb("merge_mem_with_extension.pdb")

for chain in foo.lines.keys():
    for resi in foo.lines[chain]:
        for atom in foo.lines[chain][resi]:
            print(atom.set_chain("B"))
foo.write_pdb(path='merge_mem_with_extension_2.pdb',new_lines=True)

# residue_index_list=[]
# for chain in foo.lines.keys():
#     for resi in foo.lines[chain]:
#         residue_atom_list = []
#         for atom in foo.lines[chain][resi]:
#             residue_atom_list.append(atom.y)
#         if sum(residue_atom_list)/len(residue_atom_list) < 52.91:
#             residue_index_list.append(resi)





