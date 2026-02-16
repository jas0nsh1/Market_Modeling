# Market Modeling

A collection of Python-based quantitative finance tools implementing derivative pricing, risk management, and stochastic modeling.

### **Technical Overview**

| Category | Files |
| :--- | :--- |
| **Pricing Models** | `Black_Scholes.py`, `Bond_Pricing.py`, `ZeroCouponBonds.py`, `OPTIONS.py` |
| **Stochastic Processes** | `GBM.py`, `Ornstein_Uhlenbeck.py`, `Wiener_Process.py`, `Vasicek.py` |
| **Risk & Statistics** | `Value_at_Risk.py`, `Normal_Dist.py`, `VaR_Monte_Carlo.py` |
| **Portfolio & Analysis** | `CAPM_Model.py`, `Markowitz_Model.py`, `Monte_Carlo_Stock.py`, `Monte_Carlo_Option.py` |

---

### **Key Features**
* **Stochastic Simulation:** Modeling asset price paths using Geometric Brownian Motion and mean-reverting processes like Ornstein-Uhlenbeck.
* **Option Valuation:** Analytical solutions via Black-Scholes and numerical approaches using Monte Carlo simulations.
* **Risk Metrics:** Implementation of Value-at-Risk (VaR) and normal distribution analysis for market volatility.
* **Portfolio Optimization:** Capital Asset Pricing Model (CAPM) and Markowitz Modern Portfolio Theory implementations.

### **Requirements**
* Python 3.x
* NumPy
* Pandas
* Matplotlib

### **Usage**
Each script is designed to be modular. You can run individual models directly to generate price simulations or calculate specific financial metrics.
