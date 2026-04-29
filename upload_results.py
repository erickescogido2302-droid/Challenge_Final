import os
# Script corregido para compatibilidad con Windows CMD
os.system("git add .")
os.system("git commit -m \"Resultados del challenge\"")
os.system("git push origin main")