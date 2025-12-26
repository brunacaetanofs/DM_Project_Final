import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df_fin = pd.read_csv('customer_with_clusters.csv') # this data includes the clustering labels and number of flights
df_fin['NumFlights'] = df_fin['NumFlights']/6 # annualize the number of flights (data is for 6 years)
df_fin = df_fin.dropna(subset=['merged_labels']) # remove the outliers that were not clustered

df_fin['merged_labels'] = df_fin['merged_labels'].astype(int) 

# ==============================================================================
# 1. FINANCIAL PROFILING DEFINITION
# ==============================================================================

def get_financial_profile(cluster_id):
    """
    Financial mapping based on the cluster profiling:
    Cluster 0: Seasonal (Once-a-Year) - High ticket price (peak season), medium risk.
    Cluster 1: Emerging (High Potential) - Needs investment to grow.
    Cluster 2: Core Flyers (Baseline/Volume) - The 'Cash Cow', high volume, stable.
    Cluster 3: Drifters (Low Engagement) - Lowest value, high churn.
    Cluster 4: Steady Loyalists (Long Term) - Stable, low maintenance cost.
    """
    
    # Cluster 2: "Core Flyers" (Operational Backbone)
    if cluster_id == 2: 
        return pd.Series({
            'Avg_Ticket': 250,      # Solid value, frequent flying
            'Margin': 0.15,         # Good margin
            'Retention_Years': 6,   # Long-term retention
            'Mkt_Cost': 45             # Low cost (already familiar with brand)
        })
    
    # Cluster 1: "Emerging Momentum" (High Potential)
    elif cluster_id == 1: 
        return pd.Series({
            'Avg_Ticket': 180,      # Still growing
            'Margin': 0.12, 
            'Retention_Years': 4,   # High potential to stay
            'Mkt_Cost': 65          # Worth investing to convert to Core
        })
    
    # Cluster 4: "Steady Loyalists" (Low Risk)
    elif cluster_id == 4: 
        return pd.Series({
            'Avg_Ticket': 150,      # Average value
            'Margin': 0.10, 
            'Retention_Years': 8,   # Very high retention
            'Mkt_Cost': 35           # Low maintenance
        })

    # Cluster 0: "Once-a-Year Traditionalists" (Seasonal)
    elif cluster_id == 0: 
        return pd.Series({
            'Avg_Ticket': 300,      # Expensive tickets (Peak Season travel)
            'Margin': 0.18,         # High margin due to seasonal pricing
            'Retention_Years': 2,   # Intermittent behavior
            'Mkt_Cost': 20           # Cost to reactivate during specific seasons
        })
    
    # Cluster 3: "Low-Engagement Drifters" (Low Value)
    else: # cluster_id == 3
        return pd.Series({
            'Avg_Ticket': 80,       # Cheap/short flights
            'Margin': 0.05,         # Minimal margin
            'Retention_Years': 1,   # High churn risk
            'Mkt_Cost': 0           # Do not invest
        })

# Apply the profile to each customer
financial_data = df_fin['merged_labels'].apply(get_financial_profile)
df_fin = pd.concat([df_fin, financial_data], axis=1)

# --- CALCULATE METRICS (Revenue, Profit, CLV) ---
df_fin['Annual_Revenue'] = df_fin['NumFlights'] * df_fin['Avg_Ticket']
df_fin['Annual_Profit'] = df_fin['Annual_Revenue'] * df_fin['Margin']
df_fin['CLV'] = df_fin['Annual_Profit'] * df_fin['Retention_Years']

# Display summary table
print("\n--- Financial Estimates by Cluster ---")
summary_table = df_fin.groupby('merged_labels')[['Annual_Revenue', 'Annual_Profit', 'CLV']].mean()
print(summary_table.round(2))


# ==============================================================================
# 2. ROI ANALYSIS (CAMPAIGN SIMULATION)
# ==============================================================================

# Scenario: Marketing Campaign targeting a 15% Lift in Revenue
LIFT = 0.15

# Group by Cluster to analyze total impact
roi_analysis = df_fin.groupby('merged_labels').agg({
    'Annual_Revenue': 'sum',
    'Mkt_Cost': 'sum',   # Total cost if we contact everyone in the cluster
    'merged_labels': 'count'   # Number of customers
}).rename(columns={'merged_labels': 'Num_Customers'})
# Projections
roi_analysis['Projected_Revenue'] = roi_analysis['Annual_Revenue'] * (1 + LIFT)
# Assuming the margin on the EXTRA revenue is 10% (conservative estimate)
roi_analysis['Profit_Gain'] = (roi_analysis['Projected_Revenue'] - roi_analysis['Annual_Revenue']) * 0.10 

# ROI Calculation: (Gain - Investment) / Investment
roi_analysis['ROI (%)'] = ((roi_analysis['Profit_Gain'] - roi_analysis['Mkt_Cost']) / roi_analysis['Mkt_Cost']) * 100

print("\n--- Campaign ROI Projection ---")
print(roi_analysis.round(2))


# ==============================================================================
# 3. VISUALIZATIONS (FOR THE REPORT)
# ==============================================================================

sns.set_style("whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# --- PLOT 1: Average CLV ---
sns.barplot(x=summary_table.index, y='CLV', data=summary_table, ax=axes[0], palette='viridis')
axes[0].set_title('CLV by Segment', fontsize=14)
axes[0].set_ylabel('Avg CLV (€)')
axes[0].set_xlabel('Cluster (0=Season, 1=Emergent, 2=Core, 3=Drift, 4=Loyal)')

# --- PLOT 2: Campaign ROI ---
# Color logic: Red for negative ROI, Green for positive
colors = ['red' if x < 0 else 'green' for x in roi_analysis['ROI (%)']]
sns.barplot(x=roi_analysis.index, y='ROI (%)', data=roi_analysis, ax=axes[1], palette=colors)
axes[1].set_title('Projected ROI of Marketing Campaign', fontsize=14)
axes[1].set_ylabel('ROI (%)')
axes[1].axhline(0, color='black', linestyle='--') # Break-even line

# --- PLOT 3: Investment Timeline (Focus on Core Flyers - Cluster 2) ---
# We focus on Cluster 2 ("Core Flyers") as the financial engine
target_cluster = 2 

if target_cluster in roi_analysis.index:
    total_investment = roi_analysis.loc[target_cluster, 'Mkt_Cost']
    total_gain = roi_analysis.loc[target_cluster, 'Profit_Gain']

    months = range(0, 13)
    cash_flow = []

    for m in months:
        if m == 0:
            # Month 0: Initial investment (Cash Out)
            cash_flow.append(-total_investment)
        else:
            # Revenue comes in gradually over the year
            monthly_gain = total_gain / 12
            cash_flow.append(cash_flow[-1] + monthly_gain)

    axes[2].plot(months, cash_flow, marker='o', color='blue', linewidth=2, label=f'Core Flyers (Clust {target_cluster})')
    axes[2].axhline(0, color='red', linestyle='--', label='Break-even')
    axes[2].set_title(f'Investment Timeline: Core Flyers Campaign', fontsize=14)
    axes[2].set_xlabel('Months since Campaign Launch')
    axes[2].set_ylabel('Cumulative Profit (€)')
    axes[2].legend()
    # Shade area for visual impact
    axes[2].fill_between(months, 0, cash_flow, where=[c > 0 for c in cash_flow], color='green', alpha=0.1)
    axes[2].fill_between(months, 0, cash_flow, where=[c < 0 for c in cash_flow], color='red', alpha=0.1)

# ==============================================================================
# 5. GLOBAL COST-BENEFIT ANALYSIS (TOTAL STRATEGY)
# ==============================================================================

# Vamos somar apenas os clusters onde decidimos investir (Ignoramos o Cluster 3 se o custo for 0)
# Ou somamos tudo se o custo do Cluster 3 for 0, não afeta a matemática.

total_investment = roi_analysis['Mkt_Cost'].sum()
total_gain = roi_analysis['Profit_Gain'].sum()
net_benefit = total_gain - total_investment
global_roi = (net_benefit / total_investment) * 100

print("\n================================================")
print("   FINAL COST-BENEFIT ANALYSIS (GLOBAL)   ")
print("================================================")
print(f"Total Investment Required:   € {total_investment:,.2f}")
print(f"Total Projected Profit Gain: € {total_gain:,.2f}")
print(f"Net Benefit (Profit - Cost): € {net_benefit:,.2f}")
print(f"Global Project ROI:          {global_roi:.2f}%")
print("================================================")

plt.tight_layout()
plt.show()