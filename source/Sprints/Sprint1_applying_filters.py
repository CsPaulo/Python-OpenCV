import cv2
import os

input_directory = "Dataset_leafs/test_filters/images"
output_directory = "Dataset_leafs/test_filters/processed"

# Criar as pastas para cada filtro
filter_folders = ["media", "mediana", "gaussian", "equalized", "clahe", "original_images"]
for folder in filter_folders:
    folder_path = os.path.join(output_directory, folder)
    os.makedirs(folder_path, exist_ok=True)

image_files = os.listdir(input_directory)

# Aplicar filtros em cada imagem
for image_file in image_files:
    image_path = os.path.join(input_directory, image_file)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Filtros
    image_media = cv2.blur(image, (5, 5)) 
    image_mediana = cv2.medianBlur(image, 5)  
    image_gaussian = cv2.GaussianBlur(image, (5, 5), 0)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_equalized = cv2.equalizeHist(gray_image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image_clahe = clahe.apply(gray_image)
    
    # Exibir e salvar imagens processadas
    filters = [
        ("media", image_media),
        ("mediana", image_mediana),
        ("gaussian", image_gaussian),
        ("equalized", image_equalized),
        ("clahe", image_clahe)
    ]
    
    for filter_name, filtered_image in filters:
        # Exibir imagem
        cv2.imshow(f"Imagem com {filter_name.capitalize()}", filtered_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Salvar imagem
        output_path = os.path.join(output_directory, filter_name, f"{filter_name}_{image_file}")
        cv2.imwrite(output_path, filtered_image)

print("Processamento concluído.")