 AI-Based Dynamic Resource Allocation for Reducing Blocking Probability
 Overview
This project implements an AI-driven network resource allocation system that optimizes frequency slot assignment to reduce data blocking rates in high-traffic networks. The system uses Machine Learning (Random Forest) to predict connection holding times and optimize slot allocation dynamically.

 Steps Implemented
1- Simulating Network Traffic Data

Generated 1000 synthetic network packets with random arrival times, source-destination pairs, required frequency slots, and holding times.
This simulates a high-traffic elastic optical network (EON) where efficient slot allocation is critical.
2- Training an AI Model for Resource Allocation

Used Random Forest Regressor to predict holding time based on:
Simulation time
Source & Destination nodes
Frequency slots requested
Split data into training (80%) and testing (20%) sets.
3- Predicting and Evaluating AI-Based Optimization

The model predicts how long each connection will hold resources, allowing dynamic slot preallocation to minimize blocking probability.
Evaluated performance using Mean Absolute Error (MAE) to measure prediction accuracy.
4- Analyzing Feature Importance

Determined which parameters have the greatest impact on holding time optimization (e.g., time of request, source-destination pair, slot requirements).
5- Comparing Holding Time Before and After AI Optimization

Before: Static allocation with higher congestion and blocking probability.
After: AI-optimized slot preallocation, reducing congestion and improving resource efficiency.
6- Visualizing Key Insights

Feature Importance Graph: Shows which parameters influence holding time and slot allocation the most.
Optimization Impact Chart: Compares before vs. after optimization holding times to highlight improvements.

 Results and Findings
AI-based allocation reduces blocking probability by dynamically predicting and optimizing resource usage.
Most influential features include slot requests, time of simulation, and source-destination pairs, which should be prioritized in network routing.
Potential Future Enhancements:
   Implement Deep Learning (DNN or LSTM) models for real-time adaptive learning.
 Apply Reinforcement Learning (e.g., Q-Learning, DQN) for continuous slot allocation optimization.
 Conclusion:
This AI-driven approach significantly enhances network efficiency, reducing data blocking probability and ensuring smarter bandwidth allocation in high-traffic environments. 🚀
