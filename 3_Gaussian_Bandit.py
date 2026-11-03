import streamlit as st
import numpy as np
from bandit_utils import initialize_gaussian_bandits, gaussian_thompson_sampling_step

st.title("Gaussian Thompson Sampling Bandit")

st.markdown("Run a multi-round simulation of Gaussian Thompson Sampling with 3 alternatives.")

# Inputs
col1, col2 = st.columns(2)
with col1:
    true_means_str = st.text_input("True Means (comma-separated)", "1, 2, 3")
    sigma = st.number_input("Known Noise Standard Deviation (sigma)", min_value=0.01, value=1.0, step=0.1)
with col2:
    rounds = st.number_input("Number of Rounds", min_value=1, value=100, step=10)
    seed = st.number_input("Random Seed", value=42, step=1)

run_button = st.button("Run Simulation")

if run_button:
    # Validation
    try:
        true_means = [float(x.strip()) for x in true_means_str.split(",")]
        if len(true_means) != 3:
            st.error("Please provide exactly 3 true means.")
            st.stop()
    except ValueError:
        st.error("Invalid format for true means. Use comma-separated numbers (e.g., 1, 2, 3).")
        st.stop()
        
    if sigma <= 0:
        st.error("Known standard deviation sigma must be positive (> 0).")
        st.stop()
        
    if rounds <= 0:
        st.error("Number of rounds must be a positive integer.")
        st.stop()

    # Run simulation loop
    rng = np.random.default_rng(seed=int(seed))
    posteriors = initialize_gaussian_bandits(n_arms=3, prior_mean=0.0, prior_var=25.0)
    
    selection_counts = [0, 0, 0]
    total_observed_reward = 0.0
    
    for _ in range(int(rounds)):
        chosen_arm, reward = gaussian_thompson_sampling_step(posteriors, true_means, sigma, rng)
        selection_counts[chosen_arm] += 1
        total_observed_reward += reward
        
    # Recommendation: alternative with largest posterior mean
    posterior_means = [arm["mean"] for arm in posteriors]
    recommended_arm = int(np.argmax(posterior_means))
    
    st.success("Simulation complete!")
    
    # Display results
    st.subheader("Experiment Summary")
    st.metric("Total Observed Reward", f"{total_observed_reward:.2f}")
    st.metric("Recommended Alternative", f"Alternative {recommended_arm + 1} (True Mean: {true_means[recommended_arm]})")
    
    st.subheader("Alternative Details")
    for i in range(3):
        st.markdown(f"""
        - **Alternative {i+1}**:
            - True Mean: `{true_means[i]}`
            - Posterior Estimated Mean: `{posterior_means[i]:.4f}`
            - Selection Count: `{selection_counts[i]}` ({(selection_counts[i]/rounds)*100:.1f}%)
        """)
