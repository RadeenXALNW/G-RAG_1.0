def merge_files(file1, file2, output_file):
    with open(file1, 'r') as f1, open(file2, 'r') as f2, open(output_file, 'w') as out:
        out.write(f1.read() + f2.read())

# Usage
merge_files('/teamspace/studios/this_studio/NeurIPS_Materials/pipeline_output/final_output_2.txt', '/teamspace/studios/this_studio/definitions.txt', 'final_output2xdefinition.txt')