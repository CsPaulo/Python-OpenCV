from PIL import Image
import os
import glob

source_directory = 'Dataset_leafs/Anthracnose'

destination_directory = 'Dataset_leafs_resize/Anthracnose'

new_size = (256, 256)

# Percorre todas as imagens no diretório de origem usando glob
images = glob.glob(os.path.join(source_directory, '*.jpg'))

for image_path in images:
    try:
        image = Image.open(image_path)
        
        resized_image = image.resize(new_size)
        
        file_name = os.path.basename(image_path)
        
        # Cria o diretório de destino se não existir
        if not os.path.exists(destination_directory):
            os.makedirs(destination_directory)
        
        # Salva a imagem redimensionada no diretório de destino
        resized_image.save(os.path.join(destination_directory, file_name))
        
        print(f'Imagem {file_name} redimensionada e salva com sucesso.')
    
    except Exception as e:
        print(f'Erro ao processar a imagem {image_path}: {str(e)}')
