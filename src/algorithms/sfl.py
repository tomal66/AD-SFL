def run_sfl_round(clients, server):
    """
    Executes a single communication round of the SplitFed (SFL V1) algorithm.
    
    1. All clients perform forward propagation up to the cut layer in parallel.
    2. Server receives smashed data from all clients.
    3. Server performs forward and backward propagation for all clients.
    4. All clients receive gradients and perform backward propagation in parallel.
    5. Client models are aggregated using FedAvg and re-broadcast to clients.
    """
    
    smashed_activations_list = []
    labels_list = []
    
    # Step 1: Client Forward Propagation
    # In a real distributed system, this happens concurrently on devices.
    # Here we loop over them simulating parallel execution.
    for client in clients:
        # Each client gets its next batch and does a forward pass
        smashed_data, labels = client.forward_pass()
        smashed_activations_list.append((client.id, smashed_data))
        labels_list.append((client.id, labels))
        
    total_loss = 0.0
    total_acc = 0.0
    grad_to_clients_map = {}
    
    # Step 2 & 3: Server Forward and Backward Propagation
    # The server processes the smashed data from each client.
    for client_id, smashed_data in smashed_activations_list:
        # Find the corresponding labels
        labels = next(l for cid, l in labels_list if cid == client_id)
        
        # Server trains on this client's batch
        grad_to_client, loss, acc = server.train_step(smashed_data, labels)
        
        grad_to_clients_map[client_id] = grad_to_client
        total_loss += loss
        total_acc += acc
        
    avg_loss = total_loss / len(clients)
    avg_acc = total_acc / len(clients)
    
    # Step 4: Client Backward Propagation
    # Clients independently finish the backprop using the gradients from the server.
    for client in clients:
        grad = grad_to_clients_map[client.id]
        client.backward_pass(grad)
        
    # Step 5: FedAvg of Client Models
    client_weights = [c.get_weights() for c in clients]
    global_weights = server.aggregate_client_models(client_weights)
    
    # Broadcast updated client model back to all clients
    for client in clients:
        client.set_weights(global_weights)
        
    return avg_loss, avg_acc
