import os
notes = []
files = os.listdir("notes")

for file_name in files:

    with open(f"notes/{file_name}","r") as file:
        content = file.read()
        notes.append(content)
print(notes)