import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import datetime
import uuid
from datetime import timedelta
from statsmodels.stats.power import TTestIndPower
import io
from openpyxl import Workbook

# Set page config for wider layout
st.set_page_config(layout="wide")

# Custom CSS for prominent selectbox
st.markdown("""
<style>
    div[data-testid="stSelectbox"] {
        background-color: #f0f2f6;
        border-radius: 4px;
        padding: 10px;
        margin-bottom: 20px;
    }
    div[data-testid="stSelectbox"] label {
        font-size: 18px;
        font-weight: bold;
        color: #333;
        text-align: center;
    }
    div[data-testid="stSelectbox"] select {
        background-color: #0066cc;
        color: white;
        border-radius: 4px;
        padding: 8px;
        font-size: 16px;
        width: 100%;
        max-width: 300px;
        margin: 0 auto;
        display: block;
    }
</style>
""", unsafe_allow_html=True)

# Function to generate synthetic customer data (vectorized for performance)
def generate_customer_data(start_date, end_date, monthly_signups):
    days = (end_date - start_date).days + 1
    daily_signups = int(monthly_signups / 30)
    
    # Generate dates and signup counts
    dates = [start_date + timedelta(days=i) for i in range(days)]
    signup_counts = [
        int(daily_signups * (0.7 if d.weekday() >= 5 else 1.0) * np.random.uniform(0.9, 1.1))
        for d in dates
    ]
    
    # Get current date for capping conversion times
    current_date = datetime.datetime.now()
    
    # Generate customer data
    data = []
    for date, count in zip(dates, signup_counts):
        for _ in range(count):
            signup_time = date + timedelta(
                hours=np.random.randint(0, 24),
                minutes=np.random.randint(0, 60),
                seconds=np.random.randint(0, 60)
            )
            conversion_rate = np.random.uniform(0.2, 0.3)
            conversion_time = (
                signup_time + timedelta(days=int(np.random.exponential(scale=15)))
                if np.random.random() < conversion_rate else None
            )
            # Ensure conversion time is not in the future
            if conversion_time and conversion_time > current_date:
                conversion_time = None
            
            data.append({
                "customer_id": str(uuid.uuid4()),
                "signup_timestamp": signup_time,
                "conversion_timestamp": conversion_time
            })
    
    df = pd.DataFrame(data)
    df.columns = [x.upper() for x in df.columns]
    df['SIGNUP_DATE'] = pd.to_datetime(df['SIGNUP_TIMESTAMP'])
    df['CONVERSION_DATE'] = pd.to_datetime(df['CONVERSION_TIMESTAMP'], errors='coerce')
    df['DAYS_TO_CONVERSION'] = (df['CONVERSION_DATE'] - df['SIGNUP_DATE']).dt.days
    df.drop(columns=['SIGNUP_TIMESTAMP', 'CONVERSION_TIMESTAMP'], inplace=True)
    return df

# Power analysis function
def power_analysis_for_2_sample_proportions_ztest(
    DAILY_SIGNUP_THROUGHPUT,
    BASELINE,
    CURE_DAYS,
    EXPERIMENT_START_DATE,
    ALPHA_LIST,
    POWER_LIST,
    LIFT_LIST,
    TAIL_LIST,
    METRIC_NAME,
    BFC=1,
):
    EXPERIMENT_START_DATE = pd.to_datetime(EXPERIMENT_START_DATE)
    PANDAS_LIST = []
    for k in LIFT_LIST:
        for i in ALPHA_LIST:
            for j in POWER_LIST:
                for l in TAIL_LIST:
                    HYPOTHESIZED = (1 + k) * BASELINE
                    if k < 0:
                        cohen_h_effect_size = -(2 * np.arcsin(np.sqrt(BASELINE)) - 2 * np.arcsin(np.sqrt(HYPOTHESIZED)))
                        DIFFERENCE = BASELINE - HYPOTHESIZED
                    else:
                        cohen_h_effect_size = 2 * np.arcsin(np.sqrt(HYPOTHESIZED)) - 2 * np.arcsin(np.sqrt(BASELINE))
                        DIFFERENCE = HYPOTHESIZED - BASELINE

                    power_analysis = TTestIndPower()
                    SAMPLE_SIZE = power_analysis.solve_power(
                        effect_size=cohen_h_effect_size,
                        power=j,
                        nobs1=None,
                        ratio=1,
                        alpha=i / BFC,
                        alternative=l
                    )
                    DATA_COLLECTION_END_DATE = EXPERIMENT_START_DATE + timedelta(days=int(np.ceil(SAMPLE_SIZE / (DAILY_SIGNUP_THROUGHPUT / 2))))
                    METRIC_CURE_DATE = DATA_COLLECTION_END_DATE + timedelta(days=CURE_DAYS)

                    DATA_COLLECTION_END_DATE = DATA_COLLECTION_END_DATE.strftime("%Y-%m-%d")
                    METRIC_CURE_DATE = METRIC_CURE_DATE.strftime("%Y-%m-%d")
                    PANDAS_LIST.append([
                        1 - i,
                        j,
                        l.upper(),
                        round(BASELINE, 4),
                        round(HYPOTHESIZED, 4),
                        round(DIFFERENCE, 4),
                        round(k, 4),
                        round(cohen_h_effect_size, 2),
                        round(SAMPLE_SIZE),
                        int(DAILY_SIGNUP_THROUGHPUT / 2),
                        int(np.ceil(SAMPLE_SIZE / int(DAILY_SIGNUP_THROUGHPUT / 2))),
                        DATA_COLLECTION_END_DATE,
                        METRIC_CURE_DATE
                    ])
    return pd.DataFrame(PANDAS_LIST, columns=[
        'Statistical Significance',
        'Statistical Power',
        'Test Type',
        f'Baseline {METRIC_NAME}',
        f'Hypothesized {METRIC_NAME}',
        'Difference Between Groups (Percent Pts)',
        'Difference Between Groups (Percent Change) (MDE)',
        'Standardized Effect Size (Cohen\'s H)',
        'Required Sample Size per Test Group',
        'Number of Customers Expected to Enter Test Group per Day',
        'Days Required to Achieve Sample Size',
        'Data Collection End Date',
        'Metric Cure Date'
    ])

# Power analysis summary function
def generate_power_analysis_summary(selected_row, metric_name, daily_throughput, start_date, power_analysis_start_date, power_analysis_end_date, bfc=1):
    if selected_row['Test Type'] == 'TWO-SIDED':
        test_type = 'Two-Tailed Test (evaluating for an effect in both directions)'
    else:
        test_type = 'One-Tailed Test (evaluating for an effect in one direction)'
    
    bfc_correction = ('No sample size adjustment was made for multiple comparisons.'
                     if bfc == 1 else f'Sample size adjusted for a Bonferroni Correction Factor of {bfc}.')
    
    summary = f"""
- Baseline {metric_name}: {round(selected_row[f'Baseline {metric_name}'] * 100, 3):,.3f}%
- Hypothesized {metric_name}: {round(selected_row[f'Hypothesized {metric_name}'] * 100, 3):,.3f}%
- Difference between hypothesized and baseline: {round((selected_row[f'Hypothesized {metric_name}'] - selected_row[f'Baseline {metric_name}']) * 100, 3):,.3f}%
- Minimum Detectable Effect (smallest detectable change in the metric): {selected_row['Difference Between Groups (Percent Change) (MDE)'] * 100:,.1f}%
- Required observations: {selected_row['Required Sample Size per Test Group']:,.0f} per group
- Historical median daily throughput: {int(daily_throughput):,.0f} observations/day
- Estimated days required: {selected_row['Days Required to Achieve Sample Size']:,.0f}
- Planned experiment start date: {start_date}
- Estimated data collection end date: {selected_row['Data Collection End Date']}
- Data fully cured by: {selected_row['Metric Cure Date']}
- Other Testing Parameters:
  - Type 1 Error Probability (chance of false positive): {round((1 - selected_row['Statistical Significance']) * 100, 1):,.1f}%
  - Type 2 Error Probability (chance of missing a true effect): {round((1 - selected_row['Statistical Power']) * 100, 1):,.1f}%
  - Test Type: {test_type}
  - {bfc_correction}
- Power analysis based on customers who signed up between {power_analysis_start_date} and {power_analysis_end_date}
"""
    return summary

# Visualization function for signups (using Plotly)
def visualize_signups(df, start_range, end_range):
    """Visualizes customer signups over time with a selectable date range."""
    plot_df = df.copy()
    plot_df['SIGNUP_DATE'] = pd.to_datetime(plot_df['SIGNUP_DATE']).dt.date
    plot_df = plot_df.groupby('SIGNUP_DATE')['CUSTOMER_ID'].count().reset_index()
    plot_df = plot_df[(plot_df['SIGNUP_DATE'] >= start_range) & (plot_df['SIGNUP_DATE'] <= end_range)]
    fig = px.line(
        plot_df,
        x='SIGNUP_DATE',
        y='CUSTOMER_ID',
        title='Daily Signups Over Time',
        labels={'SIGNUP_DATE': 'Signup Date', 'CUSTOMER_ID': 'Daily Signups'}
    )
    fig.update_layout(xaxis_tickangle=45)
    return fig

# Visualization function for conversion curve
def visualize_conversion_curve(df, start_range, end_range, days_input):
    """Visualizes the cumulative proportion of conversions with a single vertical line."""
    plot_df = df[df['CONVERSION_DATE'].notnull()].copy()
    plot_df = plot_df[
        (plot_df['SIGNUP_DATE'].dt.date >= start_range) &
        (plot_df['SIGNUP_DATE'].dt.date <= end_range)
    ]
    
    if plot_df.empty:
        return None
    
    agg_df = plot_df.groupby('DAYS_TO_CONVERSION')['CUSTOMER_ID'].count().reset_index()
    agg_df['TOTAL'] = agg_df['CUSTOMER_ID'].sum()
    agg_df['PROPORTION_OF_TOTAL_CONVERSIONS'] = agg_df['CUSTOMER_ID'] / agg_df['TOTAL']
    agg_df['CUMSUM_PROPORTION'] = agg_df['PROPORTION_OF_TOTAL_CONVERSIONS'].cumsum()
    
    proportion_at_days = agg_df[agg_df['DAYS_TO_CONVERSION'] == days_input]['CUMSUM_PROPORTION']
    proportion_value = round(proportion_at_days.iloc[0] * 100, 1) if not proportion_at_days.empty else 0
    
    fig = px.line(
        agg_df,
        x='DAYS_TO_CONVERSION',
        y='CUMSUM_PROPORTION',
        title=f'Distribution of Days Between Signup and Conversion (Data as of {pd.to_datetime("now").strftime("%Y-%m-%d")})',
        labels={
            'DAYS_TO_CONVERSION': 'Days Between Signup and Conversion',
            'CUMSUM_PROPORTION': 'Proportion of All Customers Who Converted'
        }
    )
    
    # Add vertical line that stops at the curve
    if not proportion_at_days.empty:
        fig.add_shape(
            type="line",
            x0=days_input,
            y0=0,
            x1=days_input,
            y1=proportion_at_days.iloc[0],
            line=dict(color="blue", dash="dash")
        )
        fig.add_annotation(
            x=days_input,
            y=proportion_at_days.iloc[0],
            text=f'{days_input} Days ({proportion_value}%)',
            showarrow=False,
            yshift=-10,
            xshift=50,
            font=dict(color="blue")
        )
    
    return fig

# Streamlit app
st.title("Power Analysis Toolkit")

st.markdown("""
This application enables users to import a dataset (or create a synthetic one for learning purposes) and conduct a power analysis using that dataset or using manual inputs.  This app also allows you to evaluate customer signup volumes over time and conversion timing which are important considerations when planning an experiment where a customer is the unit of observation and conversion is the key metric.  There are three main views:
- **Signup Visualization**: Shows daily signup trends over time, helping to ensure historical target audience throughput is indicative of future throughput.  This is important when estimating how long it will take to obtain a given number of observations in an experiment.
- **Conversion Visualization**: Displays how long it takes customers to convert after signing up, crucial for understanding how much of the total conversion that a conversion rate of a given timeframe (e.g., 30 days) used in an experiment will likely capture. This is an important consideration when balancing the comprehensiveness of the conversion metric with the amount of time it will take to have the fully cured data which is needed for the final analysis. 
- **Power Analysis**: Calculates the sample size needed for experiments to detect meaningful changes in conversion rates, ensuring reliable results.  This tab also allows you to evaluate how sample size changes with different type 1 and 2 risk levels, minimum detectable effect sizes, and test type (e.g., one-tailed vs two-tailed).  This section also creates a summary of a selected test sizing option and the assocaited inputs.
""")

# Initialize session state
if "df" not in st.session_state:
    st.session_state.df = None
if "power_analysis_df" not in st.session_state:
    st.session_state.power_analysis_df = None
if "power_analysis_days" not in st.session_state:
    st.session_state.power_analysis_days = None
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Signup Visualization"
if "analysis_mode" not in st.session_state:
    st.session_state.analysis_mode = "Dataset-Based Analysis"
if "days_input" not in st.session_state:
    st.session_state.days_input = 7  # Default value

# Define tab names
tab_names = ["Signup Visualization", "Conversion Visualization", "Power Analysis"]

# Data source selection
data_source = st.radio("Select data source:", ("Generate synthetic data (Used to Illustrate functionality of App)", "Upload CSV"))

if data_source == "Generate synthetic data (Used to Illustrate functionality of App)":
    start_date_str = st.text_input("Start Date (YYYY-MM-DD):", "2024-01-01")
    end_date_str = st.text_input("End Date (YYYY-MM-DD):", "2025-03-30")
    monthly_signups = st.number_input("Monthly Signups:", min_value=1, value=50000)

    try:
        start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date = datetime.datetime.strptime(end_date_str, "%Y-%m-%d")
        if start_date >= end_date:
            st.error("End date must be after start date.")
            st.stop()
    except ValueError:
        st.error("Invalid date format. Please use YYYY-MM-DD.")
        st.stop()

    if st.button("Generate Data"):
        with st.spinner("Generating data..."):
            st.session_state.df = generate_customer_data(start_date, end_date, monthly_signups)
            # Clear power analysis data
            st.session_state.power_analysis_df = None
            st.session_state.power_analysis_days = None
            st.session_state.days_input = 7  # Reset to default
        st.success("Data generated successfully!")

if data_source == "Upload CSV":
    st.info("""
    CSV must contain one record per customer with the following columns:
    - 'CUSTOMER_ID' (unique identifier)
    - Either 'SIGNUP_TIMESTAMP' and 'CONVERSION_TIMESTAMP' (date or timestamp format, e.g., YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)
      or 'SIGNUP_DATE' and 'CONVERSION_DATE' (date format, e.g., YYYY-MM-DD)
    Optional: 'DAYS_TO_CONVERSION' (numeric, days between signup and conversion)
    Note: 'CONVERSION_DATE' may contain empty or null values for customers who have not converted.
    """)
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, na_values=['', 'NaN', 'nan'])
            df.columns = [x.upper() for x in df.columns]
            
            # Check for required columns
            has_timestamp_cols = all(col in df.columns for col in ['CUSTOMER_ID', 'SIGNUP_TIMESTAMP', 'CONVERSION_TIMESTAMP'])
            has_date_cols = all(col in df.columns for col in ['CUSTOMER_ID', 'SIGNUP_DATE', 'CONVERSION_DATE'])
            
            if not (has_timestamp_cols or has_date_cols):
                st.error("CSV must contain 'CUSTOMER_ID' and either 'SIGNUP_TIMESTAMP' and 'CONVERSION_TIMESTAMP' or 'SIGNUP_DATE' and 'CONVERSION_DATE'.")
                st.stop()
                
            if has_timestamp_cols:
                # Process timestamp-based columns
                df['SIGNUP_DATE'] = pd.to_datetime(df['SIGNUP_TIMESTAMP'], errors='coerce')
                df['CONVERSION_DATE'] = pd.to_datetime(df['CONVERSION_TIMESTAMP'], errors='coerce')
                df.drop(columns=['SIGNUP_TIMESTAMP', 'CONVERSION_TIMESTAMP'], inplace=True)
            elif has_date_cols:
                # Process date-based columns
                df['SIGNUP_DATE'] = pd.to_datetime(df['SIGNUP_DATE'], errors='coerce')
                df['CONVERSION_DATE'] = pd.to_datetime(df['CONVERSION_DATE'], errors='coerce')
            
            # Validate SIGNUP_DATE, allow CONVERSION_DATE to be null
            if df['SIGNUP_DATE'].isnull().any():
                st.error("Invalid or missing entries in 'SIGNUP_DATE' or 'SIGNUP_TIMESTAMP'. Ensure all signup entries are valid dates or timestamps.")
                st.stop()
                
            # Calculate or retain DAYS_TO_CONVERSION
            if 'DAYS_TO_CONVERSION' not in df.columns:
                df['DAYS_TO_CONVERSION'] = (df['CONVERSION_DATE'] - df['SIGNUP_DATE']).dt.days
            else:
                df['DAYS_TO_CONVERSION'] = pd.to_numeric(df['DAYS_TO_CONVERSION'], errors='coerce')
            
            st.session_state.df = df
            # Clear power analysis data
            st.session_state.power_analysis_df = None
            st.session_state.power_analysis_days = None
            st.session_state.days_input = 7  # Reset to default
            st.success("CSV uploaded successfully!")
        except Exception as e:
            st.error(f"Error processing CSV: {e}")
            st.stop()

# Visualization section
if st.session_state.df is not None and not st.session_state.df.empty:
    signup_dates = pd.to_datetime(st.session_state.df['SIGNUP_DATE'])
    min_date = signup_dates.min().date()
    max_date = signup_dates.max().date()
    
    # Callback to preserve active tab during slider interaction
    def on_slider_change():
        st.query_params["tab"] = st.session_state.active_tab

    start_range, end_range = st.slider(
        "Select date range:",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        format="YYYY-MM-DD",
        key="date_slider",
        on_change=on_slider_change
    )

    # Display filtered sample dataset
    filtered_df = st.session_state.df[
        (st.session_state.df['SIGNUP_DATE'].dt.date >= start_range) &
        (st.session_state.df['SIGNUP_DATE'].dt.date <= end_range)
    ].copy()
    st.write("Sample of filtered dataset (first 5 rows):")
    st.dataframe(filtered_df.head(5), use_container_width=True)

    # Dropdown for tab selection (moved below dataset sample)
    st.session_state.active_tab = st.selectbox(
        "Select View:",
        tab_names,
        index=tab_names.index(st.session_state.active_tab),
        key="tab_selector"
    )
    st.query_params["tab"] = st.session_state.active_tab

    # Render tab contents
    if st.session_state.active_tab == "Signup Visualization":
        fig_signups = visualize_signups(st.session_state.df, start_range, end_range)
        st.plotly_chart(fig_signups, use_container_width=True)
    
    elif st.session_state.active_tab == "Conversion Visualization":
        st.markdown("""
        This tab allows you to evaluate the proportion of total conversions for a given cohort of customers that occurs within a certain number of days of the customer signing up. This helps determine an appropriate curing timeframe to capture a sufficient amount of the total conversion. The cohort used for evaluating conversion rate in this tab should be earlier than that which is used in the power analysis tab. This is because in this tab, we need to be able to assess conversion that occurs over a long timeframe to determine an appropriate curing timeframe; however, for the power analysis, we are more interested in getting an accurate estimation of signup throughput.
        """)
        max_days = int(st.session_state.df['DAYS_TO_CONVERSION'].max()) if st.session_state.df['DAYS_TO_CONVERSION'].notnull().any() else 45
        
        def on_days_input_change():
            st.session_state.active_tab = "Conversion Visualization"
            st.query_params["tab"] = "Conversion Visualization"
        
        st.session_state.days_input = st.number_input(
            "Days between signup and conversion:",
            min_value=0,
            max_value=max_days,
            value=st.session_state.days_input,
            step=1,
            key="conversion_days_input",
            on_change=on_days_input_change
        )
        
        # Clear power analysis if days_input changes
        if st.session_state.power_analysis_days is not None and st.session_state.power_analysis_days != st.session_state.days_input:
            st.session_state.power_analysis_df = None
            st.session_state.power_analysis_days = None
        
        fig_conversion = visualize_conversion_curve(st.session_state.df, start_range, end_range, st.session_state.days_input)
        if fig_conversion:
            st.plotly_chart(fig_conversion, use_container_width=True)
        else:
            st.warning("No conversion data available for the selected date range.")
    
    elif st.session_state.active_tab == "Power Analysis":
        st.subheader("Power Analysis Configuration")
        st.markdown("""
        This section allows you to configure the power analysis in two ways:
        - **Dataset-Based Analysis**: Uses historical data from the uploaded or generated dataset, the selected date range, and the conversion timeframe from the Conversion Visualization tab to estimate daily signups and baseline conversion rate. This is ideal when you have reliable historical data.
        - **Manual Input Analysis**: Allows you to manually specify the estimated daily signups, baseline conversion rate, and conversion timeframe. This is useful when you don't have a dataset or want to test hypothetical scenarios.
        """)
        
        def on_analysis_mode_change():
            st.session_state.active_tab = "Power Analysis"
            st.session_state.analysis_mode = st.session_state.analysis_mode_radio
            st.query_params["tab"] = "Power Analysis"
        
        analysis_mode = st.radio(
            "Select Power Analysis Mode:",
            ("Dataset-Based Analysis", "Manual Input Analysis"),
            index=0 if st.session_state.analysis_mode == "Dataset-Based Analysis" else 1,
            key="analysis_mode_radio",
            on_change=on_analysis_mode_change
        )
        st.session_state.analysis_mode = analysis_mode
        
        # Mode-specific inputs
        if analysis_mode == "Dataset-Based Analysis":
            # Filter data for power analysis
            df_target = st.session_state.df[
                (st.session_state.df['SIGNUP_DATE'].dt.date >= start_range) &
                (st.session_state.df['SIGNUP_DATE'].dt.date <= end_range)
            ].copy()
            
            # Calculate median signups per day
            signup_counts = df_target.groupby(df_target['SIGNUP_DATE'].dt.date)['CUSTOMER_ID'].count()
            median_signups_per_day = signup_counts.median()
            
            # Calculate conversion rate for specified days
            df_target['CONVERSION'] = df_target['DAYS_TO_CONVERSION'].notnull() & (df_target['DAYS_TO_CONVERSION'] <= st.session_state.days_input)
            conversion_rate = df_target['CONVERSION'].mean() * 100
            
            # Display summary statement
            st.markdown(f"""
            ### Dataset-Based Inputs
            - Median number of signups per day for customers who signed up between {start_range.strftime('%Y-%m-%d')} and {end_range.strftime('%Y-%m-%d')}: {int(median_signups_per_day):,.0f}
            - {st.session_state.days_input}-day conversion rate for this cohort (taken from Conversion Visualization tab): {conversion_rate:.1f}%
            """)
            
            # Check if cohort is too recent for conversion window
            current_date = pd.to_datetime("now").date()
            if (current_date - end_range).days < st.session_state.days_input:
                st.warning(
                    f"The selected cohort includes signups as recent as {end_range.strftime('%Y-%m-%d')}. "
                    f"With a {st.session_state.days_input}-day conversion window, the conversion metric is not fully cured "
                    f"for all customers in this cohort. The power analysis results may be affected."
                )
            
            # Set power analysis parameters
            daily_throughput = median_signups_per_day
            baseline = conversion_rate / 100
            cure_days = st.session_state.days_input
            power_analysis_start_date = start_range.strftime("%Y-%m-%d")
            power_analysis_end_date = end_range.strftime("%Y-%m-%d")
        
        else:  # Manual Input Analysis
            st.subheader("Manual Input Parameters")
            daily_throughput = st.number_input("Estimated Daily Signups:", min_value=1, value=1000, step=1)
            baseline_percent = st.number_input("Baseline Conversion Rate (%):", min_value=0.0, max_value=100.0, value=25.0, step=0.1)
            cure_days = st.number_input("Conversion Timeframe (days):", min_value=1, value=7, step=1)
            baseline = baseline_percent / 100
            power_analysis_start_date = "N/A"
            power_analysis_end_date = "N/A"
        
        # Common power analysis inputs
        st.subheader("Power Analysis Parameters")
        st.markdown("""
        **Input Explanations**:
        - **Statistical Significance Levels**: How confident you want to be that a detected effect is real (not due to chance). Common values are 0.05 (95% confidence) or 0.1 (90% confidence). Lower values require larger samples.
        - **Statistical Power Levels**: The probability of detecting a true effect if it exists. Common values are 0.8 (80% chance) or 0.9 (90% chance). Higher power requires larger samples.
        - **Minimum Detectable Effects (MDE)**: The smallest change in conversion rate you want to detect (e.g., 0.05 means a 5% change). Smaller effects require larger samples.
        - **Test Type**: Two-sided tests check for any change (increase or decrease), while one-sided tests check for a change in one direction (increase if MDE is positive, decrease if negative).
        - **Bonferroni Correction Factor**: Used when running multiple related tests. Enter the number of tests (e.g., 2 if testing two metrics). Increases sample size to maintain reliability.
        - **Experiment Start Date**: When the experiment will begin.
        - **Maximum Acceptable Test Duration**: Optional limit on how long the experiment can run. Leave blank for no limit.
        - **Experiment Results Deadline**: Optional date by which results must be ready, including the conversion window.
        """)
        
        significance_input = st.text_input("Statistical Significance Levels (comma-separated, e.g., 0.05,0.1):", "0.05,0.1")
        power_input = st.text_input("Statistical Power Levels (comma-separated, e.g., 0.8,0.9):", "0.8")
        mde_input = st.text_input("Minimum Detectable Effects (comma-separated, e.g., 0.03,0.05,0.1):", "0.03,0.05,0.1")
        test_type = st.selectbox("Test Type:", ["Two-sided", "One-sided"])
        bfc_input = st.number_input("Bonferroni Correction Factor (number of related tests, e.g., 1 for no correction):", min_value=1, value=1, step=1)
        experiment_start_date = st.date_input("Experiment Start Date:", value=pd.to_datetime("2025-04-18").date())
        max_test_duration = st.text_input("Maximum Acceptable Test Duration (days, optional):", "", help="Leave blank for no maximum duration")
        results_deadline = st.date_input("Experiment Results Deadline (optional):", value=None, help="Select a date if results must be ready by a specific deadline")
        
        # Process common inputs
        try:
            alpha_list = [float(x) for x in significance_input.split(",") if x.strip()]
            power_list = [float(x) for x in power_input.split(",") if x.strip()]
            lift_list = [float(x) for x in mde_input.split(",") if x.strip()]
            tail_list = ['two-sided' if test_type == "Two-sided" else 'two-sided']
            bfc = int(bfc_input)
            max_test_duration = float(max_test_duration) if max_test_duration.strip() else 0
        except ValueError:
            st.error("Please enter valid numeric values for significance, power, MDE, and test duration (if provided).")
            st.stop()
        
        # Run power analysis
        if st.button("Run Power Analysis"):
            with st.spinner("Running power analysis..."):
                master_df = power_analysis_for_2_sample_proportions_ztest(
                    DAILY_SIGNUP_THROUGHPUT=daily_throughput,
                    BASELINE=baseline,
                    CURE_DAYS=cure_days,
                    EXPERIMENT_START_DATE=experiment_start_date.strftime("%Y-%m-%d"),
                    ALPHA_LIST=alpha_list,
                    POWER_LIST=power_list,
                    LIFT_LIST=lift_list,
                    TAIL_LIST=tail_list,
                    METRIC_NAME=f"{cure_days}-Day Conversion Rate",
                    BFC=bfc
                )
                
                # Filter by maximum test duration if specified
                if max_test_duration > 0:
                    master_df = master_df[master_df['Days Required to Achieve Sample Size'] <= max_test_duration]
                
                # Filter by results deadline if specified
                if results_deadline:
                    master_df = master_df[pd.to_datetime(master_df['Metric Cure Date']) <= pd.to_datetime(results_deadline)]
                
                master_df = master_df.sort_values('Days Required to Achieve Sample Size', ascending=False).reset_index(drop=True)
                st.session_state.power_analysis_df = master_df
                st.session_state.power_analysis_days = cure_days
                
                if master_df.empty:
                    st.warning(
                        f"No testing options found where the required sample size can be collected within the specified constraints."
                    )
                else:
                    st.write("Power Analysis Results:")
                    st.dataframe(master_df, use_container_width=True)
                    
                    # Excel download button
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        master_df.to_excel(writer, sheet_name=f'{cure_days}-Day Conversion', index=False)
                    excel_data = output.getvalue()
                    st.download_button(
                        label="Download Power Analysis as Excel",
                        data=excel_data,
                        file_name="Experiment_Sizing_Options.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
        
        # Power analysis summary
        if st.session_state.power_analysis_df is not None and not st.session_state.power_analysis_df.empty:
            if st.session_state.power_analysis_days == (st.session_state.days_input if analysis_mode == "Dataset-Based Analysis" else cure_days):
                st.subheader("Power Analysis Summary")
                selected_index = st.selectbox("Select a power analysis result:", 
                                            options=range(len(st.session_state.power_analysis_df)),
                                            format_func=lambda x: f"Option {x+1}: {st.session_state.power_analysis_df.iloc[x]['Days Required to Achieve Sample Size']} days, "
                                                               f"{st.session_state.power_analysis_df.iloc[x]['Statistical Significance']*100}% significance, "
                                                               f"{st.session_state.power_analysis_df.iloc[x]['Statistical Power']*100}% power")
                
                selected_row = st.session_state.power_analysis_df.iloc[selected_index]
                summary = generate_power_analysis_summary(
                    selected_row=selected_row,
                    metric_name=f"{st.session_state.power_analysis_days}-Day Conversion Rate",
                    daily_throughput=daily_throughput,
                    start_date=experiment_start_date.strftime("%Y-%m-%d"),
                    power_analysis_start_date=power_analysis_start_date,
                    power_analysis_end_date=power_analysis_end_date,
                    bfc=bfc
                )
                st.markdown(summary)
            else:
                st.warning(
                    f"The current power analysis results are based on a {st.session_state.power_analysis_days}-day conversion timeframe, "
                    f"but the selected conversion timeframe is {st.session_state.days_input if analysis_mode == 'Dataset-Based Analysis' else cure_days} days. "
                    f"Please rerun the power analysis to update the results."
                )

    # Download button for data with formatted dates
    df_for_download = st.session_state.df.copy()
    df_for_download['SIGNUP_DATE'] = df_for_download['SIGNUP_DATE'].dt.strftime('%Y-%m-%d')
    df_for_download['CONVERSION_DATE'] = df_for_download['CONVERSION_DATE'].dt.strftime('%Y-%m-%d')
    csv = df_for_download.to_csv(index=False, na_rep='').encode('utf-8')
    st.download_button(
        "Download Data as CSV",
        csv,
        "customer_data.csv",
        "text/csv"
    )

# Reset button
if st.button("Reset Data"):
    st.session_state.df = None
    st.session_state.power_analysis_df = None
    st.session_state.power_analysis_days = None
    st.session_state.active_tab = "Signup Visualization"
    st.session_state.analysis_mode = "Dataset-Based Analysis"
    st.session_state.days_input = 7
    st.query_params["tab"] = "Signup Visualization"
    st.success("Data reset successfully!")
