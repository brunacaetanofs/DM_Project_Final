# Customer Cluster Dashboard – Bonus Interactive Visualization

This document provides the necessary instructions to run the **Bonus Interactive Customer Cluster Dashboard** and details its main features and architecture.

LINK to demonstration of interactive visualization: https://youtu.be/FDur9wqpiYw

## Project Structure

The application is organized modularly for better maintenance and readability. The file structure is as follows:

**project_folder/**
* **app_dashboard.py**: Dash application logic (layout and callbacks).
* **DasboardRunApp.py**: Server initialization script.
* **customers_with_clusters_final.csv**: Pre-processed dataset containing the clusters.
* **BONUS.md**: User guide (this file).

---

## Requirements and Installation

Before running the dashboard, ensure your environment meets the following requirements:

1. **Python**: Version 3.8 or higher.
2. **Python Packages**: Install the required dependencies via terminal:
   ```bash
   pip install pandas dash plotly
   ```
3. **Data**: Ensure the file `customers_with_clusters_final.csv` is located in the same folder as the code files.

---

## How to Run

Follow these steps to launch the dashboard:

1. Open your terminal or command prompt in the project folder.
2. Run the initialization script:
   ```bash
   python DasboardRunApp.py
   ```
3. The terminal will indicate that the server is active. Open your browser and go to:
   **http://127.0.0.1:8050/**

---

## Dashboard Features

### 1. Interactive Filters
The dashboard allows real-time data segmentation through a sidebar:
* **Demographics**: Gender, Education, and Marital Status.
* **Geographic**: State/Province.
* **Financial**: Income Slider.

### 2. Main Visualizations

| Chart | Description |
| :--- | :--- |
| **3D Scatter Plot** | Spatial exploration of **NumFlights**, **PointsAccumulated**, and **CLV**, colored by cluster. |
| **Cluster Map** | Geographic distribution of segments by region (State/Province). |
| **Bar Chart** | Total count of customers per cluster for volume analysis. |
| **CLV Histogram** | Distribution of Customer Lifetime Value to compare profitability across clusters. |

### 3. Data Export
* **Download Filtered CSV Button**: Allows you to instantly export the currently filtered data into a CSV file.

---

## Important Notes

* **Reactivity**: All visualizations and metrics update automatically whenever a filter is changed.
* **Visual Consistency**: Cluster colors are consistent across all charts to allow for quick visual identification.
* **Filter Handling**: If a combination of filters returns no data, the charts will appear empty to prevent rendering errors.

---

**Authors:**
* **Group 23** – Data Mining Course NOVA IMS – MSc in Data Science & Business Analytics