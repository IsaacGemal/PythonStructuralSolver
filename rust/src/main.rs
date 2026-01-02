use nalgebra::{DMatrix, DVector};

fn main() {
    // Data for Elements and Nodes (1-based indexing in data)
    let info = vec![
        vec![1, 2, 3, 4],
        vec![1, 2, 9, 10],
        vec![3, 4, 9, 10],
        vec![3, 4, 5, 6],
        vec![5, 6, 7, 8],
        vec![5, 6, 9, 10],
        vec![7, 8, 9, 10],
        vec![7, 8, 11, 12],
        vec![9, 10, 11, 12],
    ];

    let ele = vec![
        vec![1.0, 0.0, 0.0, 10.0, 20.0],
        vec![2.0, 0.0, 0.0, 20.0, 10.0],
        vec![3.0, 10.0, 20.0, 20.0, 10.0],
        vec![4.0, 16.0, 20.0, 20.0, 20.0],
        vec![5.0, 20.0, 20.0, 30.0, 20.0],
        vec![6.0, 20.0, 20.0, 20.0, 10.0],
        vec![7.0, 30.0, 20.0, 20.0, 10.0],
        vec![8.0, 30.0, 20.0, 40.0, 0.0],
        vec![9.0, 20.0, 10.0, 40.0, 0.0],
    ];

    // Supports (1-based indexing)
    let supports = [1, 2, 12];

    // Forces at Nodes (1-based indexing, value)
    let forces = vec![(7, 20.0), (10, -50.0)];

    let a = 10.0; // in^2
    let e = 29000.0; // ksi

    // Preparing the stiffness matrix
    let enum_count = info.len();
    let max_dof = info.iter().flatten().max().unwrap();
    let mut k_global = DMatrix::<f64>::zeros(*max_dof, *max_dof);

    // Setting up forces
    let mut f_global = DVector::<f64>::zeros(*max_dof);
    for (dof, force_val) in &forces {
        f_global[dof - 1] = *force_val;
    }

    // Element stiffness calculations
    for i in 0..enum_count {
        let dx = ele[i][3] - ele[i][1];
        let dy = ele[i][4] - ele[i][2];
        let c = f64::sqrt(dy * dy + dx * dx);
        let dx_norm = dx / c;
        let dy_norm = dy / c;

        // Transformation matrix T
        let t = DMatrix::from_row_slice(
            2,
            4,
            &[dx_norm, dy_norm, 0.0, 0.0, 0.0, 0.0, dx_norm, dy_norm],
        );

        // Element stiffness in local coordinates
        let kee = DMatrix::from_row_slice(2, 2, &[1.0, -1.0, -1.0, 1.0]) * ((a * e) / (c * 12.0));

        // Transform to global coordinates
        let ke = t.transpose() * kee * t;

        // Assembly into the global stiffness matrix
        let indices: Vec<usize> = info[i].iter().map(|&x| x - 1).collect();
        for (local_i, &global_i) in indices.iter().enumerate() {
            for (local_j, &global_j) in indices.iter().enumerate() {
                k_global[(global_i, global_j)] += ke[(local_i, local_j)];
            }
        }
    }

    // Applying boundary conditions - create reduced system
    let support_indices: Vec<usize> = supports.iter().map(|&x| x - 1).collect();
    let free_dofs: Vec<usize> = (0..*max_dof)
        .filter(|i| !support_indices.contains(i))
        .collect();

    let n_free = free_dofs.len();
    let mut k_mod = DMatrix::<f64>::zeros(n_free, n_free);
    let mut f_mod = DVector::<f64>::zeros(n_free);

    for (i, &dof_i) in free_dofs.iter().enumerate() {
        f_mod[i] = f_global[dof_i];
        for (j, &dof_j) in free_dofs.iter().enumerate() {
            k_mod[(i, j)] = k_global[(dof_i, dof_j)];
        }
    }

    // Solving for displacements
    let displacements = k_mod.lu().solve(&f_mod).expect("Linear solve failed");

    // Initialize the full displacement array with zeros
    let mut full_displacements = DVector::<f64>::zeros(*max_dof);
    for (i, &dof) in free_dofs.iter().enumerate() {
        full_displacements[dof] = displacements[i];
    }

    // Calculating reactions
    let mut reactions = DVector::<f64>::zeros(supports.len());
    for (i, &support_dof) in supports.iter().enumerate() {
        let mut reaction = 0.0;
        for j in 0..*max_dof {
            reaction += k_global[(support_dof - 1, j)] * full_displacements[j];
        }
        reactions[i] = reaction;
    }

    // Output results
    println!("Displacements:");
    print!("[");
    for i in 0..*max_dof {
        if i > 0 {
            print!(" ");
        }
        print!("{:11.8}", full_displacements[i]);
    }
    println!("]");

    println!("Reactions:");
    print!("[");
    for i in 0..reactions.len() {
        if i > 0 {
            print!(" ");
        }
        print!("{:.1}", reactions[i]);
    }
    println!("]");
}
