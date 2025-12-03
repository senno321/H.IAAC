import sys
import json
import torch
import utils

# Adiciona a pasta utils ao path para podermos importar os módulos
sys.path.append('./') 

def get_model_params(model_name, input_shape, num_classes):
    """Carrega um modelo e conta seus parâmetros."""
    print(f"Carregando modelo: {model_name}")
    model_path = f"./model/{model_name}.pth"
    try:
        model = ModelPersistence.load(model_path, model_name, input_shape=input_shape, num_classes=num_classes)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Modelo {model_name} carregado. Parâmetros: {params}")
        return params
    except Exception as e:
        print(f"Erro ao carregar {model_name}: {e}")
        print(f"Certifique-se de ter executado 'python gen_sim_model.py' para {model_name} primeiro.")
        return None

def create_scaled_profile():
    input_shape_cifar = (3, 32, 32)
    num_classes_cifar = 10

    # 1. Calcular parâmetros
    params_simplecnn = get_model_params("simplecnn", input_shape_cifar, num_classes_cifar)
    params_shufflenet = get_model_params("Shufflenet_v2_x0_5", input_shape_cifar, num_classes_cifar)

    if params_simplecnn is None or params_shufflenet is None:
        print("Não foi possível calcular os parâmetros. Abortando.")
        return

    # 2. Calcular a proporção (ratio)
    ratio = params_simplecnn / params_shufflenet
    print(f"\nProporção (SimpleCNN / ShuffleNet): {params_simplecnn} / {params_shufflenet} = {ratio:.6f}")

    # 3. Carregar o perfil base da ShuffleNet
    shufflenet_profile_path = "./utils/profile/Shufflenet_v2_x0_5.json"
    print(f"Carregando perfil base: {shufflenet_profile_path}")
    with open(shufflenet_profile_path, 'r') as f:
        shufflenet_profile = json.load(f)

    # 4. Criar o novo perfil escalonado
    scaled_profile = {}
    for device, specs in shufflenet_profile.items():
        new_specs = specs.copy()

        # Aplicar a proporção ao custo de energia e tempo
        original_mJ = new_specs.get("training_mJ", 0)
        original_ms = new_specs.get("training_ms", 0)

        new_specs["training_mJ"] = original_mJ * ratio
        new_specs["training_ms"] = original_ms * ratio

        scaled_profile[device] = new_specs

    # 5. Salvar o novo perfil
    output_path = "./utils/profile/simplecnn_scaled.json"
    with open(output_path, 'w') as f:
        json.dump(scaled_profile, f, indent=4)

    print(f"\nPerfil proporcional salvo em: {output_path}")

if __name__ == "__main__":
    create_scaled_profile()