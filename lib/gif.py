import os
import re
from PIL import Image

def generate_gif(path_images, path_gif):
    
    pattern = re.compile(r'(error|solution)_(\d+)_(\d+)\.png$')

    immagini_per_tipo = {'error': [],'solution': []}

    for nome_file in os.listdir(path_images):
        match = pattern.match(nome_file)
        if match:
            tipo = match.group(1)
            steps = int(match.group(2))
            id_val = int(match.group(3))
            immagini_per_tipo[tipo].append((id_val, nome_file))

    for tipo, lista_immagini in immagini_per_tipo.items():

        lista_immagini.sort(key=lambda x: x[0])
        
        immagini = [Image.open(os.path_images.join(path_images, nome_file)).convert('RGB') 
                    for _, nome_file in lista_immagini]
        
        if immagini:
            output_filename = os.path_images.join(path_gif,f'{tipo}_{steps}.gif')
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