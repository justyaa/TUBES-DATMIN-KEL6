import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Page config
st.set_page_config(page_title="Adidas Sales Analysis", layout="wide")

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv('predicted_sales.csv')

data = load_data()

# Sidebar filters
with st.sidebar:
    st.header("Filter Data")
    
    # Region filter
    regions = [col.replace('Region_', '') for col in data.columns if col.startswith('Region_')]
    selected_regions = st.multiselect("Select Region(s)", regions, default=regions)
    
    # State filter
    states = [col.replace('State_', '') for col in data.columns if col.startswith('State_')]
    selected_states = st.multiselect("Select State(s)", states, default=states)
    
    # Product filter
    products = [col.replace('Product_', '') for col in data.columns if col.startswith('Product_')]
    selected_products = st.multiselect("Select Product(s)", products, default=products)

# Filter data based on selections
filtered_data = data.copy()
for region in regions:
    if region not in selected_regions:
        filtered_data = filtered_data[~filtered_data[f'Region_{region}']]
        
for state in states:
    if state not in selected_states:
        filtered_data = filtered_data[~filtered_data[f'State_{state}']]
        
for product in products:
    if product not in selected_products:
        filtered_data = filtered_data[~filtered_data[f'Product_{product}']]

# Main title
st.title("Adidas Sales Analysis Dashboard")

# Create tabs
tabs = st.tabs([
    "1. Analisis Regional", 
    "2. Analisis Produk per Region", 
    "3. Performa Keseluruhan",
    "4. Prediksi Penjualan"
])

# Tab 1: Analisis Regional
with tabs[0]:
    st.header("Analisis Regional")
    
    # Regional performance
    regional_sales = pd.DataFrame()
    for region in selected_regions:
        region_data = filtered_data[filtered_data[f'Region_{region}']]
        if not region_data.empty:
            regional_sales.loc[region, 'Total Sales'] = region_data['Predicted Total Sales'].sum()
            regional_sales.loc[region, 'Average Price'] = region_data['Price per Unit'].mean()
            regional_sales.loc[region, 'Total Units'] = region_data['Units Sold'].sum()
    
    if not regional_sales.empty:
        col1, col2, col3 = st.columns(3)
        
        # Show summary metrics
        with col1:
            total_sales = regional_sales['Total Sales'].sum()
            st.metric("Total Sales", f"${total_sales:,.2f}")
        
        with col2:
            avg_price = regional_sales['Average Price'].mean()
            st.metric("Average Price", f"${avg_price:,.2f}")
        
        with col3:
            total_units = regional_sales['Total Units'].sum()
            st.metric("Total Units Sold", f"{total_units:,}")
        
        # Show sales bar chart
        fig_regional = px.bar(regional_sales.reset_index(), 
                              x='index', y='Total Sales',
                              title='Total Penjualan per Region')
        st.plotly_chart(fig_regional)
    else:
        st.write("No data available for the selected regions.")


# Tab 2: Analisis Produk per Region
with tabs[1]:
    st.header("Analisis Produk per Region")
    
    # Create analysis for each region
    for region in selected_regions:
        st.subheader(f"Region: {region}")
        
        # Filter data for current region
        region_data = filtered_data[filtered_data[f'Region_{region}']]
        
        # Create product analysis for this region
        product_sales = pd.DataFrame()
        for product in products:
            product_data = region_data[region_data[f'Product_{product}']]
            if not product_data.empty:
                product_sales.loc[product, 'Total Sales'] = product_data['Predicted Total Sales'].sum()
                product_sales.loc[product, 'Average Price'] = product_data['Price per Unit'].mean()
                product_sales.loc[product, 'Total Units'] = product_data['Units Sold'].sum()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sales by product
            fig_product_sales = px.bar(product_sales.reset_index(), 
                                     x='index', y='Total Sales',
                                     title=f'Total Penjualan per Produk - {region}')
            st.plotly_chart(fig_product_sales)
        
        with col2:
            # Units sold by product
            fig_product_units = px.bar(product_sales.reset_index(), 
                                     x='index', y='Total Units',
                                     title=f'Total Unit Terjual per Produk - {region}')
            st.plotly_chart(fig_product_units)
        
        # Create metrics for this region
        col1, col2, col3 = st.columns(3)
        
        with col1:
            top_product = product_sales['Total Sales'].idxmax()
            st.metric(
                "Produk Terlaris",
                top_product,
                f"${product_sales.loc[top_product, 'Total Sales']:,.2f}"
            )
        
        with col2:
            avg_price = product_sales['Average Price'].mean()
            st.metric(
                "Rata-rata Harga",
                f"${avg_price:,.2f}"
            )
        
        with col3:
            total_units = product_sales['Total Units'].sum()
            st.metric(
                "Total Unit Terjual",
                f"{total_units:,}"
            )
        
        # Add comparison chart
        fig_comparison = go.Figure()
        
        # Add bars for sales
        fig_comparison.add_trace(go.Bar(
            name='Total Sales ($)',
            x=product_sales.index,
            y=product_sales['Total Sales'],
            yaxis='y',
            offsetgroup=1
        ))
        
        # Add line for average price
        fig_comparison.add_trace(go.Scatter(
            name='Average Price ($)',
            x=product_sales.index,
            y=product_sales['Average Price'],
            yaxis='y2'
        ))
        
        # Update layout
        fig_comparison.update_layout(
            title=f'Perbandingan Penjualan dan Harga per Produk - {region}',
            yaxis=dict(title='Total Sales ($)', side='left'),
            yaxis2=dict(title='Average Price ($)', side='right', overlaying='y'),
            showlegend=True
        )
        
        st.plotly_chart(fig_comparison)
        
        st.divider()

# Tab 3: Performa Keseluruhan
with tabs[2]:
    st.header("Performa Keseluruhan")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Correlation analysis
        st.subheader("Korelasi Antar Faktor")
        numeric_cols = ['Price per Unit', 'Units Sold', 'Predicted Total Sales']
        correlation = filtered_data[numeric_cols].corr()
        
        fig_corr = plt.figure(figsize=(8, 6))
        sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
        st.pyplot(fig_corr)
    
    with col2:
        # Feature importance
        st.subheader("Tingkat Kepentingan Faktor")
        X = filtered_data[['Price per Unit', 'Units Sold']]
        y = filtered_data['Predicted Total Sales']
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        importance_df = pd.DataFrame({
            'Factor': ['Harga per Unit', 'Unit Terjual'],
            'Importance': model.feature_importances_
        })
        
        fig_importance = px.bar(importance_df, x='Factor', y='Importance',
                               title='Pengaruh Faktor terhadap Total Penjualan')
        st.plotly_chart(fig_importance)

# Tab 4: Prediksi Penjualan
with tabs[3]:
    st.header("Model Prediksi Penjualan")
    
    # Prepare data for prediction
    X = filtered_data[['Price per Unit', 'Units Sold']]
    y = filtered_data['Predicted Total Sales']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Model metrics
    col1, col2 = st.columns(2)
    
    with col1:
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        st.metric("Root Mean Square Error (RMSE)", f"${rmse:.2f}")
    
    with col2:
        r2 = r2_score(y_test, y_pred)
        st.metric("R² Score", f"{r2:.3f}")
    
    # Actual vs Predicted plot
    fig_prediction = px.scatter(x=y_test, y=y_pred,
                               labels={'x': 'Actual Sales', 'y': 'Predicted Sales'},
                               title='Perbandingan Nilai Aktual vs Prediksi')
    fig_prediction.add_shape(type="line", line=dict(dash="dash"),
                            x0=y_test.min(), y0=y_test.min(),
                            x1=y_test.max(), y1=y_test.max())
    st.plotly_chart(fig_prediction)
    
    # Sales prediction input
    st.subheader("Simulasi Prediksi Penjualan")
    
    col1, col2 = st.columns(2)
    
    with col1:
        price = st.number_input(
            "Masukkan Harga per Unit ($)", 
            min_value=0.0, 
            value=float(filtered_data['Price per Unit'].mean()),
            step=10.0,
            format="%.2f"
        )
    
    with col2:
        units = st.number_input(
            "Masukkan Jumlah Unit", 
            min_value=0, 
            value=int(filtered_data['Units Sold'].mean()),
            step=1
        )
    
    if st.button("Prediksi Penjualan"):
        # Calculate expected baseline
        baseline = price * units
        
        # Make prediction using the model
        prediction = model.predict([[price, units]])
        
        # Add context to the prediction
        st.success(f"Prediksi Total Penjualan: ${prediction[0]:,.2f}")
        st.info(f"""
        Analisis Prediksi:
        - Harga per Unit: ${price:,.2f}
        - Jumlah Unit: {units:,}
        - Nilai Dasar (Price × Units): ${baseline:,.2f}
        """)
        
        # Add warning if prediction seems unrealistic
        if prediction[0] < baseline * 0.5 or prediction[0] > baseline * 1.5:
            st.warning("""
            ⚠️ Perhatian: Prediksi mungkin memerlukan kalibrasi model.
            Nilai prediksi berbeda signifikan dari perhitungan dasar (harga × unit).
            """)