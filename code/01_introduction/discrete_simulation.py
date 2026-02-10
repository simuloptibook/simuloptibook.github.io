import heapq, random

# Set random seed for reproducibility
random.seed(42)

# Simulation parameters
sim_end = 100  # simulation ends at time 100
num_cashiers = 3  # number of cashiers available

# Event tracking
event_list = []
# Initialize the first arrival
first_arrival_time = random.expovariate(1/5)  # mean interarrival time = 5

# Cashier status: True if busy, False if free
cashier_busy = [False] * num_cashiers

# Queue: list of (arrival_time, customer_id)
queue = []

# Service times tracking (for debugging/analysis)
service_times = {}

# Statistics tracking
total_wait_time = 0
total_customers = 0
completed_customers = 0

# Helper functions
def next_interarrival():
    return random.expovariate(1/5)  # mean interarrival time = 5

def service_time(cid):
    # Service time depends on customer (deterministic in this example)
    if cid not in service_times:
        service_times[cid] = random.expovariate(1/8)  # mean service time = 8
    return service_times[cid]

def any_cashier_free():
    return not all(cashier_busy)

def start_checkout(cid, time):
    # Find first free cashier
    for i in range(num_cashiers):
        if not cashier_busy[i]:
            cashier_busy[i] = True
            break

def start_service(cid, time):
    # This is called when a customer is taken from queue to be served
    start_checkout(cid, time)

def enqueue(cid, time):
    queue.append((time, cid))

def dequeue():
    if queue:
        arrival_time, cid = queue.pop(0)
        return cid
    return None

def queue_not_empty():
    return len(queue) > 0

def finish_service(cid, time):
    # Find which cashier was serving this customer
    for i in range(num_cashiers):
        if cashier_busy[i]:
            # We'll just mark this cashier as free after service
            # In a more sophisticated model, we'd track which cashier serves which customer
            cashier_busy[i] = False
            break

    # Statistics
    global total_wait_time, completed_customers
    if cid in service_times:
        completed_customers += 1
    # Calculate wait time (service_time - arrival_time) could be tracked better in a real implementation

def next_id():
    # Simple counter for customer IDs
    if not hasattr(next_id, "counter"):
        next_id.counter = 1
    next_id.counter += 1
    return next_id.counter

# Visualization function
def visualize_state(time, message=""):
    """Print current state of the simulation for visualization"""
    status = f"\n--- Time: {time:.2f} --- {message}\n"
    
    # Cashier status
    status += "Cashiers: "
    for i, busy in enumerate(cashier_busy):
        status += f"[{'Busy' if busy else 'Free'}] "
    status += "\n"
    
    # Queue status
    status += f"Queue ({len(queue)} customers): "
    if queue:
        queue_info = [f"C{cid}(wait:{time-at:.1f})" for at, cid in queue[:5]]  # Show first 5
        if len(queue) > 5:
            queue_info.append(f"... +{len(queue)-5} more")
        status += ", ".join(queue_info)
    else:
        status += "Empty"
    status += "\n"
    
    print(status)

# Initialize simulation
time = 0

# First event: arrival of customer 1
heapq.heappush(event_list, (first_arrival_time, 'arrival', 1))

# Main simulation loop
while event_list and time < sim_end:
    if not event_list:
        break
        
    time, ev_type, cid = heapq.heappop(event_list)
    
    if ev_type == 'arrival':
        visualize_state(time, f"Customer {cid} arrived")
        
        if any_cashier_free():
            start_checkout(cid, time)
            heapq.heappush(event_list, (time + service_time(cid), 
                'departure', cid))
            visualize_state(time, f"Customer {cid} started checkout")
        else:
            enqueue(cid, time)
            visualize_state(time, f"Customer {cid} joined queue")
        
        # Schedule next arrival
        heapq.heappush(event_list, (time + next_interarrival(), 
            'arrival', next_id()))
            
    elif ev_type == 'departure':
        visualize_state(time, f"Customer {cid} departed")
        finish_service(cid, time)
        if queue_not_empty():
            next_cid = dequeue()
            start_service(next_cid, time)
            heapq.heappush(event_list, (time + service_time(next_cid), 
                'departure', next_cid))
            visualize_state(time, f"Customer {next_cid} started checkout from queue")

            
# Print final statistics
print(f"Completed {completed_customers} customers in {sim_end} time units")
print(f"Total queue length at end: {len(queue)}")
print(f"Cashier status: {cashier_busy}")
