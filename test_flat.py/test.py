import uproot


file_path = 'dataset/4lep 2/Data/data_A.4lep.root'

try:
    with uproot.open(file_path) as file:
        tree = file["mini"]
        print("Branches found in the file:")
       
        print(tree.keys())
except Exception as e:
    print(f"An error occurred: {e}")