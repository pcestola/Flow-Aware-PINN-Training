import os
import re
from PIL import Image

# Cartella contenente le immagini
cartella_immagini = '/raid/homespace/piecestola/space/PINN/results/heat_rectangle/run_0/images'

# Pattern per estrarre tipo, step, e id
pattern = re.compile(r'image_(\d+)_(error|solution)_(\d+)\.png$')

# Dizionario per raccogliere immagini per tipo
immagini_per_tipo = {
    'error': [],
    'solution': []
}

# Estrai le informazioni e organizza per tipo
for nome_file in os.listdir(cartella_immagini):
    match = pattern.match(nome_file)
    if match:
        step = int(match.group(1))     # ora corretto
        tipo = match.group(2)          # 'error' o 'solution'
        id_val = int(match.group(3))   # numero temporale
        immagini_per_tipo[tipo].append((id_val, nome_file))

# Crea una GIF per ciascun tipo
for tipo, lista_immagini in immagini_per_tipo.items():
    # Ordina per 'id'
    lista_immagini.sort(key=lambda x: x[0])
    
    # Carica immagini
    immagini = [Image.open(os.path.join(cartella_immagini, nome_file)).convert('RGB') 
                for _, nome_file in lista_immagini]
    
    # Crea e salva la GIF
    if immagini:
        output_filename = f'{tipo}.gif'
        immagini[0].save(
            output_filename,
            save_all=True,
            append_images=immagini[1:],
            duration=100,
            loop=0
        )
        print(f"GIF creata con successo: {output_filename}")
    else:
        print(f"Nessuna immagine trovata per tipo: {tipo}")