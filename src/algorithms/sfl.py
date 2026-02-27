def run_sfl_round(clients, server, local_epochs=1):
    """
    Executes a single communication round of the SplitFed (SFL V1) algorithm.
    A round consists of `local_epochs` complete iterations over the clients' local datasets.
    
    1. Clients iterate over their batches. For each batch:
        a. Clients perform forward propagation up to the cut layer in parallel.
        b. Server receives smashed data and performs forward/backward propagation.
        c. Clients receive gradients and perform local backward propagation.
    2. Once all clients complete their local epochs, client models are aggregated using FedAvg and re-broadcast.
    """
    
    total_loss = 0.0
    total_acc = 0.0
    total_batches = 0
    
    for epoch in range(local_epochs):
        for client in clients:
            client.reset_iterator()
            
        while True:
            smashed_activations_list = []
            labels_list = []
            active_clients = []
            
            # Step 1: Client Forward Propagation
            # In a real distributed system, this happens concurrently on devices.
            # Here we loop over them simulating parallel execution.
            for client in clients:
                result = client.forward_pass(global_round=epoch)
                if result[0] is not None:
                    smashed_data, labels = result
                    smashed_activations_list.append((client.id, smashed_data))
                    labels_list.append((client.id, labels))
                    active_clients.append(client)
            
            # If no clients have data left, this epoch is complete
            if not active_clients:
                break
                
            grad_to_clients_map = {}
            
            # Step 2 & 3: Server Forward and Backward Propagation
            # The server processes the smashed data from each client.
            for client_id, smashed_data in smashed_activations_list:
                labels = next(l for cid, l in labels_list if cid == client_id)
                grad_to_client, loss, acc = server.train_step(smashed_data, labels, client_id=client_id)
                
                grad_to_clients_map[client_id] = grad_to_client
                total_loss += loss
                total_acc += acc
                total_batches += 1
                
            # Step 4: Client Backward Propagation
            # Clients independently finish the backprop using the gradients from the server.
            for client in active_clients:
                grad = grad_to_clients_map[client.id]
                client.backward_pass(grad)
                
    avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
    avg_acc = total_acc / total_batches if total_batches > 0 else 0.0
    
    # Step 5: FedAvg of Client Models
    client_weights = [c.get_weights() for c in clients]
    global_weights = server.aggregate_client_models(client_weights)
    
    # Broadcast updated client model back to all clients
    for client in clients:
        client.set_weights(global_weights)
        
    # Step 6: FedAvg of Server Models
    server.aggregate_server_models()
        
    return avg_loss, avg_acc
