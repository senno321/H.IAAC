def get_selected_cids_and_local_training_data_size(results):
    selected_cids = {}

    for result in results:
        ds_size = result[1].num_examples
        cid = result[1].metrics["cid"]
        selected_cids[cid] = ds_size

    return selected_cids


def get_training_time_per_cid(client_profile, client_dataset_size, epochs):
    cids_time = {cid: {"proc": 0, "total": 0} for cid in client_dataset_size.keys()}

    max_training_round_time = 0
    for cid in cids_time:
        ds_size = client_dataset_size[cid]
        proc = client_profile[cid]["training_ms"] * ds_size * epochs
        cids_time[cid]["proc"] = proc
        cids_time[cid]["total"] = proc
        max_training_round_time = max(max_training_round_time, proc)

    return cids_time, max_training_round_time


def get_selected_cid_training_energy(client_profile, selected_cids_time, client_dataset_size, epochs):
    cid_joule_consumption = {cid: 0 for cid in selected_cids_time.keys()}

    for cid in selected_cids_time.keys():
        ds_size = client_dataset_size[cid]
        proc_joules = client_profile[cid]["training_mJ"] * ds_size * epochs
        cid_joule_consumption[cid] = proc_joules

    return cid_joule_consumption
