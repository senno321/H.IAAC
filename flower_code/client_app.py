import torch
from flwr.client import ClientApp
from flwr.common import Context

# from client.base import FlowerClient
from client.base import BaseClient
from client.critical import CriticalClient
from client.feddyn import FedDynClient
from client.fedcs import FedCSClient 
from utils.simulation.config import set_seed
from utils.simulation.workflow import get_user_dataloader, get_initial_model


def client_fn(context: Context):
    # 1. config
    set_seed(context.run_config["seed"])
    
    # 2. client id
    cid = context.node_config["partition-id"]
    flwr_cid = str(context.node_id)
    
    # 3. model
    model = get_initial_model(context)
    
    # 4. dataloader
    dataloader = get_user_dataloader(context, cid)
    dataset_id = context.run_config["hugginface-id"]
    
    # 5. client selection
    is_critical = context.run_config.get("is-critical", False) # Adicionei .get para segurança
    selection_name = context.run_config.get("selection-name", "random")

    # Lógica de decisão
    if is_critical:
        return CriticalClient(
            cid=cid, 
            flwr_cid=flwr_cid, 
            model=model, 
            dataloader=dataloader,
            dataset_id=dataset_id
        ).to_client()
        
    elif selection_name == "feddyn":
        return FedDynClient(
            cid=cid, 
            flwr_cid=flwr_cid, 
            model=model, 
            dataloader=dataloader,
            dataset_id=dataset_id
        ).to_client()
        
    elif selection_name == "fedcs":
        # Instancia o cliente do FedCS
        return FedCSClient(
            cid=cid, 
            flwr_cid=flwr_cid, 
            model=model, 
            dataloader=dataloader,
            dataset_id=dataset_id
        ).to_client()
        
    else:
        # Default (Random, Constant, etc)
        return BaseClient(
            cid=cid, 
            flwr_cid=flwr_cid, 
            model=model, 
            dataloader=dataloader,
            dataset_id=dataset_id
        ).to_client()


app = ClientApp(client_fn=client_fn)