import os
import re
from PIL import Image

def generate_gif(path_images, path_gif, steps):
    pattern = re.compile(r'(error|solution)_(\d+)_(\d+)\.png$')
    immagini_per_tipo = {'error': [], 'solution': []}

    for nome_file in os.listdir(path_images):
        match = pattern.match(nome_file)
        if match:
            tipo = match.group(1)
            step_val = int(match.group(2))
            id_val = int(match.group(3))

            if step_val == steps:
                immagini_per_tipo[tipo].append((id_val, nome_file))

    for tipo, lista_immagini in immagini_per_tipo.items():
        lista_immagini.sort(key=lambda x: x[0])
        immagini = []

        for _, nome_file in lista_immagini:
            img_path = os.path.join(path_images, nome_file)
            immagini.append(Image.open(img_path).convert('RGB'))

        if immagini:
            os.makedirs(path_gif, exist_ok=True)
            output_filename = os.path.join(path_gif, f'{tipo}_{steps}.gif')
            immagini[0].save(
                output_filename,
                save_all=True,
                append_images=immagini[1:],
                duration=100,
                loop=0
            )
            print(f"✅ GIF creata con successo: {output_filename}")
        else:
            print(f"⚠️ Nessuna immagine trovata per tipo: {tipo}")

        # Cancella immagini originali
        for _, nome_file in lista_immagini:
            try:
                os.remove(os.path.join(path_images, nome_file))
            except OSError as e:
                print(f"Errore nella cancellazione {nome_file}: {e}")