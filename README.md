# FGC Portfolio Rebalancer

## Overview

### Product Summary

A financial analytics and portfolio optimization tool that performs bond portfolio rebalancing using real-world data and quantitative metrics. The system evaluates securities based on factors such as duration, yield, credit rating, and risk sensitivity to construct a more balanced and optimized portfolio.

The project focuses on applying quantitative finance techniques to simulate and improve portfolio allocation decisions.

### Goal

Develop a portfolio rebalancing system that demonstrates:

quantitative finance and fixed income analysis
portfolio optimization using real financial metrics
risk management through duration and DV01
data-driven decision making for asset allocation

Portfolio rebalancing is a core concept in investment management used to maintain target asset allocations and manage risk exposure over time

## Objectives & Success Metrics

### Primary Objectives

Analyze a set of fixed income securities
Compute key financial metrics (duration, yield, DV01, rating)
Rebalance portfolio weights based on optimization criteria
Maintain a balance between risk and return

### Success Metrics

Improved portfolio balance across risk dimensions
Logical redistribution of asset weights
Consistency with financial intuition and constraints
Clear interpretability of rebalancing decisions

## Target Users

### Primary User

Recruiters and interviewers evaluating quantitative finance skills

### Secondary Use Cases

Students learning portfolio management
Individuals interested in bond portfolio optimization

## Key Features

### Data Input

Accept a set of bond securities with attributes such as:
yield
duration
credit rating
price sensitivity
Metric Computation
Calculate key financial indicators including:
Duration (interest rate sensitivity)
DV01 (dollar value of a basis point)
Yield comparison
Risk contribution
Portfolio Analysis
Evaluate current allocation across securities
Identify imbalances in risk or return exposure
Highlight overweight and underweight positions
Rebalancing Engine
Adjust portfolio weights to improve allocation
Optimize for a balance between yield and risk
Ensure constraints such as total allocation consistency
Output & Interpretation
Provide updated portfolio weights
Show before vs after allocation
Enable understanding of trade-offs in rebalancing

## Tech Stack

Language: Python
Data Processing: Pandas
Numerical Computation: NumPy
Analysis: Quantitative finance methods

## How It Works

Input a set of bond securities with relevant financial metrics
Compute derived values such as duration and DV01
Analyze the current portfolio allocation
Apply rebalancing logic to adjust weights
Output the optimized portfolio and compare results

## Installation and Setup

Clone the repository and navigate into the project directory
Install required dependencies
Run the main script or notebook
Input bond data and observe rebalancing results

## Example Output

Initial portfolio allocation
Computed metrics for each security
Rebalanced portfolio weights
Comparison between original and optimized allocations

## Future Improvements

Add support for larger and dynamic datasets
Incorporate optimization algorithms (e.g., mean-variance)
Integrate real-time financial data APIs
Add visualization dashboards
Extend to multi-asset portfolios