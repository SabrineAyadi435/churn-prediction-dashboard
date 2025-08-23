from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import traceback
import os
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')  
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from datetime import datetime

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}
app.secret_key = 'your-secret-key-here'  # Required for flash messages

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variable to store the latest predictions
latest_predictions = None
survival_model = None
survival_scaler = None
# Data cache for survival analysis

# Default output structure for survival analysis
default_output = {
    'hazard_curves': {
        'time_points': list(range(0, 73)),
        'cumulative_hazard': [0.02 * x for x in range(73)],
        'quartiles': {
            'q25': [0.01 * x for x in range(73)],
            'q50': [0.02 * x for x in range(73)],
            'q75': [0.03 * x for x in range(73)]
        }
    },
    'business_insights': {
        'critical_windows': {
            'high_risk_window': 3,
            'stable_window': 6
        },
        'intervention_points': {
            'immediate': 0,
            'urgent': 0,
            'watch': 0
        },
        'twelve_month_hazard': 0
    },
    'active_page': 'survival'
}

try:
    # Classification model
    churn_model = joblib.load('models/churn_model.pkl')
    print("✅ Churn model loaded successfully!")
    
    # Survival model - loaded directly as CoxPHFitter object
    survival_model = joblib.load('models/cox_model.pkl')
    
    # Load the scaler separately
    survival_scaler = joblib.load('models/scaler.pkl')
    
    print("✅ All models loaded successfully!")
    
except Exception as e:
    print(f"❌ Error loading models: {e}")
    print(traceback.format_exc())
    survival_model = None
    survival_scaler = None
    exit()


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def encode_features(input_df):
    """Encode all categorical features for the model"""
    df = input_df.copy()
    
    # Contract encoding
    df['Contract_One year'] = (df['Contract'] == 'One year').astype(int)
    df['Contract_Two year'] = (df['Contract'] == 'Two year').astype(int)
    
    # InternetService encoding
    df['InternetService_Fiber optic'] = (df['InternetService'] == 'Fiber optic').astype(int)
    df['InternetService_No'] = (df['InternetService'] == 'No').astype(int)
    
    # PaymentMethod encoding
    df['PaymentMethod_Credit card (automatic)'] = (df['PaymentMethod'] == 'Credit card (automatic)').astype(int)
    df['PaymentMethod_Electronic check'] = (df['PaymentMethod'] == 'Electronic check').astype(int)
    df['PaymentMethod_Mailed check'] = (df['PaymentMethod'] == 'Mailed check').astype(int)
    
    # Binary features
    binary_features = ['PaperlessBilling', 'Dependents', 'Partner', 'OnlineSecurity',
                      'OnlineBackup', 'DeviceProtection', 'TechSupport',
                      'StreamingTV', 'StreamingMovies','SeniorCitizen']
    
    for feat in binary_features:
        if feat in df.columns:
            df[feat] = (df[feat] == 'Yes').astype(int)
     
    df = df.rename(columns={'SeniorCitizen': 'SeniorCitizen_1'})
    return df

def prepare_model_input(df):
    """Prepare features for model prediction"""
    encoded_df = encode_features(df)
    
    # Ensure all expected features exist
    expected_features = churn_model.feature_names_in_
    for feature in expected_features:
        if feature not in encoded_df.columns:
            encoded_df[feature] = 0
    
    return encoded_df[expected_features]

def prepare_survival_data(input_df):
    """Prepare data for survival analysis with proper encoding and scaling"""
    if survival_model is None:
        raise ValueError("Survival model not loaded")
    
    df = input_df.copy()
    
    # Ensure required columns exist with defaults
    if 'Churn' not in df.columns:
        df['Churn'] = (df.get('churn_probability', 0) > 0.5).astype(int)
    
    # Apply categorical encoding first
    df_encoded = encode_features(df)
    
    # Create TotalCharges if missing
    if 'TotalCharges' not in df_encoded.columns:
        df_encoded['TotalCharges'] = df_encoded['MonthlyCharges'] * df_encoded['tenure']
    
    # Prepare survival-specific columns
    df_encoded['duration'] = df_encoded['tenure']
    df_encoded['event'] = df_encoded['Churn']
    
    # Drop non-predictor columns
    cols_to_drop = ['customerID', 'tenure', 'Churn', 'churn_probability']
    df_surv = df_encoded.drop(columns=[c for c in cols_to_drop if c in df_encoded.columns])
    
    # Scale numerical features - must scale BOTH together
    if survival_scaler:
        required_num_cols = ['MonthlyCharges', 'TotalCharges']
        if all(col in df_surv.columns for col in required_num_cols):
            try:
                # Create temp DataFrame with just the numeric columns
                num_data = df_surv[required_num_cols].copy()
                # Scale both columns together
                scaled_values = survival_scaler.transform(num_data)
                df_surv[required_num_cols] = scaled_values
            except Exception as e:
                app.logger.error(f"Scaling error: {str(e)}")
                # Fallback to unscaled if error occurs
        else:
            app.logger.warning(f"Cannot scale - missing required numeric columns. Need: {required_num_cols}")
    
    # Ensure all expected features exist (from survival model)
    expected_features = survival_model.params_.index.tolist()
    for feature in expected_features:
        if feature not in df_surv.columns:
            df_surv[feature] = 0
    
    # Convert all data to float (catch any remaining strings)
    try:
        df_surv = df_surv.astype(float)
    except ValueError as e:
        app.logger.error(f"Type conversion error: {str(e)}")
        # Find and fix problematic columns
        for col in df_surv.columns:
            if df_surv[col].dtype == object:
                try:
                    df_surv[col] = pd.to_numeric(df_surv[col], errors='coerce').fillna(0)
                except:
                    df_surv[col] = 0
    
    return df_surv

def analyze_results(df, predictions):
    """Analyze and segment predictions"""
    df['churn_probability'] = predictions
    
    # Create risk groups
    df['risk_group'] = pd.cut(df['churn_probability'],
                             bins=[0, 0.3, 0.7, 1],
                             labels=['Low', 'Medium', 'High'])
    
    # Generate recommendations - fixed version
    df['recommendation'] = np.where(
        df['churn_probability'] > 0.7,
        'Immediate retention call with special offer',
        np.where(
            df['churn_probability'] > 0.4,
            'Targeted email campaign with loyalty benefits',
            'Regular monitoring and satisfaction survey'
        )
    )
    
    return df

@app.route('/')
def home():
    """Redirect to overview page as the default landing page"""
    return redirect(url_for('overview'))

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    global latest_predictions
    
    
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                df = pd.read_csv(filepath)
            except Exception as e:
                flash(f'Error reading file: {str(e)}', 'error')
                return redirect(request.url)
            
            required = ['tenure', 'MonthlyCharges', 'Contract', 'InternetService', 'PaymentMethod']
            missing = [col for col in required if col not in df.columns]
            if missing:
                flash(f'Missing required columns: {missing}', 'error')
                return redirect(request.url)
            
            try:
                # Prepare features and predict
                model_input = prepare_model_input(df)
                predictions = churn_model.predict_proba(model_input)[:, 1]
                
                # Analyze results
                results_df = analyze_results(df.copy(), predictions)
                latest_predictions = results_df.to_dict(orient='records')
                
                flash('File successfully processed', 'success')
                return redirect(url_for('overview'))
            except Exception as e:
                flash(f'Error processing data: {str(e)}', 'error')
                return redirect(request.url)
    
    return render_template('upload.html', active_page='upload')
@app.route('/overview')
def overview():
    global latest_predictions

    # Initialize metrics with safe defaults
    metrics = {
        'current_churn_rate': 26.0,
        'revenue_at_risk': 140000,
        'avg_lifetime': 18,
        'at_risk_customers': 1120,
        'total_customers': 5000,
        'total_arr': 1000000,
        'revenue_change': 5.2,
    }

    insights = {
        'month_to_month_survival': 12,
        'fiber_vs_dsl_diff': 6,
        'echeck_churn_prob': 42.5
    }

    impact = {
        'projected_loss': 750000,
        'savings_potential': 325000,
        'projected_loss_breakdown': [
            {'name': 'Month-to-month', 'value': 450000},
            {'name': 'Fiber optic', 'value': 187500},
            {'name': 'E-check', 'value': 112500}
        ],
        'savings_potential_breakdown': [
            {'name': 'Top 10%', 'value': 162500},
            {'name': 'Next 15%', 'value': 97500},
            {'name': 'Others', 'value': 65000}
        ]
    }

    focus = {
        'month_to_month_at_risk': 680,
        'early_dropoff_rate': 38,
        'echeck_customers': 420
    }

    # Initialize recommendations with empty/default values
    recommendations = {
        'contract_incentive': 0,
        'contract_target': 0,
        'contract_impact': 0,
        'payment_discount': 0,
        'payment_target': 0,
        'payment_impact': 0,
        'onboarding_target': 0,
        'onboarding_impact': 0
    }

    churn_distribution = [0, 0, 0]
    survival_months = list(range(0, 37, 3))
    survival_rates = [100 * (1 - 0.26 * (t/36)) for t in survival_months]
    
    # Initialize survival-specific metrics
    intervention_points = {
        'immediate': 0,
        'urgent': 0,
        'watch': 0
    }

    if latest_predictions:
        try:
            df = pd.DataFrame(latest_predictions)
            
            if df.empty:
                flash('Uploaded data is empty', 'warning')
            else:
                # Ensure required columns exist
                required_cols = ['MonthlyCharges', 'churn_probability', 'risk_group', 'tenure']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    flash(f'Missing required columns: {missing_cols}', 'warning')
                else:
                    # Calculate metrics with zero-division protection
                    total_arr = max(df['MonthlyCharges'].sum() * 12, 1)
                    revenue_at_risk_total = (df['MonthlyCharges'] * df['churn_probability']).sum() * 12
                    
                    metrics.update({
                        'current_churn_rate': float((df['churn_probability'] > 0.5).mean() * 100),
                        'revenue_at_risk': float(revenue_at_risk_total),
                        'at_risk_customers': int(len(df[df['risk_group'].isin(['High', 'Medium'])])),
                        'total_customers': int(len(df)),
                        'total_arr': float(total_arr),
                        'revenue_change': 5.2  # Can be calculated if needed
                    })

                    # Calculate insights
                    if 'Contract' in df.columns:
                        month_to_month = df[df['Contract'] == 'Month-to-month']
                        if not month_to_month.empty:
                            insights['month_to_month_survival'] = int(month_to_month['tenure'].mean())

                    if 'InternetService' in df.columns:
                        fiber = df[df['InternetService'] == 'Fiber optic']
                        dsl = df[df['InternetService'] == 'DSL']
                        if not fiber.empty and not dsl.empty:
                            insights['fiber_vs_dsl_diff'] = int(
                                (fiber['churn_probability'].mean() - dsl['churn_probability'].mean()) * 100
                            )

                    if 'PaymentMethod' in df.columns:
                        echeck = df[df['PaymentMethod'] == 'Electronic check']
                        if not echeck.empty:
                            insights['echeck_churn_prob'] = float(echeck['churn_probability'].mean() * 100)

                    # Calculate impact
                    projected_loss = revenue_at_risk_total * 0.75
                    savings_potential = revenue_at_risk_total * 0.25
                    
                    impact.update({
                        'projected_loss': int(projected_loss),
                        'savings_potential': int(savings_potential),
                        'projected_loss_breakdown': [
                            {'name': 'Month-to-month', 'value': int(projected_loss * 0.6)},
                            {'name': 'Fiber optic', 'value': int(projected_loss * 0.25)},
                            {'name': 'E-check', 'value': int(projected_loss * 0.15)}
                        ],
                        'savings_potential_breakdown': [
                            {'name': 'Top 10%', 'value': int(savings_potential * 0.5)},
                            {'name': 'Next 15%', 'value': int(savings_potential * 0.3)},
                            {'name': 'Others', 'value': int(savings_potential * 0.2)}
                        ]
                    })

                    # Calculate focus areas
                    focus.update({
                     'month_to_month_at_risk': int(len(df[(df['Contract'] == 'Month-to-month') & (df['risk_group'] == 'High')])) if 'Contract' in df.columns else 0,
                     'early_dropoff_rate': int((len(df[(df['tenure'] < 6) & (df['risk_group'] == 'High')]) / len(df) * 100) if len(df) > 0 else 0),
                     'echeck_customers': int(len(df[df['PaymentMethod'] == 'Electronic check'])) if 'PaymentMethod' in df.columns else 0
                    })

                    # Calculate churn distribution
                    churn_distribution = [
                        int(len(df[df['risk_group'] == 'Low'])),
                        int(len(df[df['risk_group'] == 'Medium'])),
                        int(len(df[df['risk_group'] == 'High']))
                    ]

                    # Calculate survival metrics using the same approach as survival route
                    if survival_model:
                        try:
                            # Prepare survival data
                            survival_data = prepare_survival_data(df.copy())
                            
                            # Time points for prediction (0-72 months)
                            time_points = list(range(0, 73))
                            
                            # Get survival predictions
                            survival_preds = survival_model.predict_survival_function(survival_data, times=time_points)
                            
                            
                            # Calculate key survival metrics
                            survival_curve_mean = survival_preds.mean(axis=1)
                            median_survival = next((t for t, s in zip(time_points, survival_curve_mean) if s <= 0.5), time_points[-1])
                            
                            
                            # Update metrics with survival values
                            metrics.update({
                                'avg_lifetime': int(median_survival), 
                                
                            })
                            
                            # Update survival rates for the chart
                            survival_rates = [
                                float(survival_curve_mean.loc[t] * 100)
                                for t in survival_months
                                if t in survival_curve_mean.index
                            ]
                            
                        except Exception as e:
                            app.logger.error(f"Survival model error in overview: {str(e)}")
                            # Fallback to simple calculation if survival model fails
                            survival_rates = [float(100 * (1 - metrics['current_churn_rate']/100 * (t/36))) for t in survival_months]

        except Exception as e:
            app.logger.error(f"Error in overview: {str(e)}\n{traceback.format_exc()}")
            flash('An error occurred processing your data', 'error')

    return render_template(
        'overview.html',
        metrics=metrics,
        insights=insights,
        impact=impact,
        focus=focus,
        recommendations=recommendations,
        churn_distribution=churn_distribution,
        survival_months=survival_months,
        survival_rates=survival_rates,
        
        active_page='overview'
    )





@app.route('/survival')
def survival():
    global latest_predictions, survival_model
    
    if latest_predictions is None:
        flash('Please upload data first', 'warning')
        return redirect(url_for('upload'))
        
    if survival_model is None:
        flash('Survival analysis model not available', 'error')
        return render_template('survival.html', error="Model not loaded")

    try:
        df = pd.DataFrame(latest_predictions)
        total_customers = len(df)
        
        # Prepare survival data - ensure your prepare_survival_data() returns proper DataFrame
        df_surv = prepare_survival_data(df)
        
        # Time points for prediction (0-72 months)
        time_points = list(range(0, 73))  # 0 to 72 months inclusive
        
        # Get survival predictions
        survival_preds = survival_model.predict_survival_function(df_surv, times=time_points)
        
        # Convert to clean Python data structures
        survival_data = {
            'time_points': [int(t) for t in time_points],
            'survival_probability': [float(p) for p in survival_preds.mean(axis=1)],
            'quartiles': {
                'q25': [float(p) for p in survival_preds.quantile(0.25, axis=1)],
                'q75': [float(p) for p in survival_preds.quantile(0.75, axis=1)]
            },
            'milestones': {
                '3mo': {'value': 3, 'text': "Early Churn"},
                '6mo': {'value': 6, 'text': "Stabilization"},
                '12mo': {'value': 12, 'text': "Annual Retention"},
                '24mo': {'value': 24, 'text': "Long-term"}
            }
        }

        # Calculate key metrics
        median_survival = next((t for t, s in zip(time_points, survival_data['survival_probability']) if s <= 0.5), time_points[-1])
        twelve_month_survival = survival_data['survival_probability'][12] * 100
        three_month_survival = survival_data['survival_probability'][3] * 100

        # Calculate hazard rates
        hazard_preds = survival_model.predict_cumulative_hazard(df_surv, times=time_points)
        twelve_month_hazard = hazard_preds.mean(axis=1).iloc[12] * 100

        # Risk classification
        critical_months = []
        short_term_preds = survival_model.predict_survival_function(df_surv, times=[3, 6, 12])
        for i in range(len(df_surv)):
            survival_curve = short_term_preds.iloc[:, i]
            critical_month = next((t for t, s in zip([3, 6, 12], survival_curve) if s <= 0.5), 12)
            critical_months.append(critical_month)
        
        intervention_points = {
            'immediate': sum(1 for x in critical_months if x <= 3),
            'urgent': sum(1 for x in critical_months if 3 < x <= 6),
            'watch': sum(1 for x in critical_months if x > 6)
        }

        # Business insight
        business_insight = (
            f"Retention analysis shows {twelve_month_survival:.1f}% of customers remain after 1 year. "
            f"Critical risk: {intervention_points['immediate']} customers (0-3 months). "
            f"Median customer lifespan: {median_survival} months."
        )

        return render_template('survival.html',
            survival_curves=survival_data,
            business_insights={
                'median_survival': median_survival,
                'three_month_survival': three_month_survival,
                'twelve_month_survival': twelve_month_survival,
                'twelve_month_hazard': twelve_month_hazard,
                'intervention_points': intervention_points,
                'business_insight': business_insight
            },
            total_customers=total_customers
        )

    except Exception as e:
        app.logger.error(f"Survival analysis error: {str(e)}\n{traceback.format_exc()}")
        return render_template('survival.html', error=str(e))







@app.route('/classification')
def classification():
    global latest_predictions
    
    if latest_predictions is None:
        flash('Please upload data first', 'warning')
        return redirect(url_for('upload'))
    
    df = pd.DataFrame(latest_predictions)
    model_input = prepare_model_input(df)
    
    # Feature Importance
    feature_names = churn_model.feature_names_in_
    coefficients = churn_model.coef_[0]
    feature_importance = dict(zip(feature_names, abs(coefficients)))
    
    # Business Insights
    try:
        # Prepare data for visualizations
        risk_group_data = df['risk_group'].value_counts().reset_index()
        risk_group_data.columns = ['risk_group', 'count']
        
        monthly_charges_data = df.groupby('risk_group')['MonthlyCharges'].mean().reset_index()
        
        service_dist_data = pd.crosstab(df['risk_group'], df['InternetService']).reset_index()
        
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        top_features_df = pd.DataFrame(top_features, columns=['feature', 'importance'])
        
        business_insights = {
            'top_churn_drivers': top_features[:3],
            'high_risk_profile': {
                'avg_tenure': df[df['risk_group']=='High']['tenure'].mean(),
                'avg_charge': df[df['risk_group']=='High']['MonthlyCharges'].mean(),
                'common_services': {
                    'internet': df[df['risk_group']=='High']['InternetService'].mode()[0],
                    'contract': df[df['risk_group']=='High']['Contract'].mode()[0]
                }
            },
            'revenue_at_risk': df[df['risk_group']=='High']['MonthlyCharges'].sum(),
            'plotly_data': {
                'risk_groups': risk_group_data.to_dict('records'),
                'monthly_charges': monthly_charges_data.to_dict('records'),
                'service_distribution': service_dist_data.to_dict('records'),
                'top_features': top_features_df.to_dict('records')
            }
        }
    except Exception as e:
        flash(f'Business insights error: {str(e)}', 'warning')
        business_insights = None

    classification_details = {
        'risk_groups': df['risk_group'].value_counts().to_dict(),
        'recommendations': df['recommendation'].value_counts().to_dict(),
        'feature_importance': feature_importance,
        'business_insights': business_insights
    }
    
    return render_template('classification.html', 
                         details=classification_details,
                         active_page='classification',
                         predictions=latest_predictions)
@app.route('/recommendations')
def recommendations():
    global latest_predictions
    
    if latest_predictions is None:
        flash('Please upload data first', 'warning')
        return redirect(url_for('upload'))
    
    try:
        # Convert to DataFrame with proper defaults
        df = pd.DataFrame(latest_predictions)
        
        # Ensure required columns exist
        if 'customerID' not in df.columns:
            df['customerID'] = [f'cust-{i+1:04d}' for i in range(len(df))]
        
        required_cols = ['tenure', 'MonthlyCharges', 'Contract', 'InternetService', 'PaymentMethod']
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0 if col == 'tenure' else 'Unknown'
        
        # Calculate TotalCharges if missing
        if 'TotalCharges' not in df.columns:
            df['TotalCharges'] = df['MonthlyCharges'] * df['tenure']
        
        # Get classification predictions
        model_input = prepare_model_input(df)
        df['churn_probability'] = churn_model.predict_proba(model_input)[:, 1]
        df['risk_level'] = np.where(
            df['churn_probability'] > 0.7, 'High',
            np.where(df['churn_probability'] > 0.4, 'Medium', 'Low'))
        
        # Get survival predictions
        df['critical_month'] = 72  # Default value
        if survival_model:
            try:
                survival_data = prepare_survival_data(df.copy())
                survival_preds = survival_model.predict_survival_function(survival_data)
                time_points = survival_preds.index
                
                for i in range(len(df)):
                    try:
                        survival_curve = survival_preds.iloc[:, i]
                        below_half = survival_curve[survival_curve < 0.5]
                        df.at[i, 'critical_month'] = below_half.index[0] if not below_half.empty else time_points[-1]
                    except:
                        df.at[i, 'critical_month'] = max(6, min(72, 72 * (1 - df.at[i, 'churn_probability'])))
            except Exception as e:
                app.logger.error(f"Survival prediction error: {str(e)}")
                df['critical_month'] = np.clip(72 * (1 - df['churn_probability']), 6, 72)
        
        # Calculate survival probability at 12 months
        df['12mo_survival'] = np.clip(1 - (df['churn_probability'] * (12/72)), 0, 1)
        # Calculate survival probability at 12 months using the survival model
        df['12mo_survival'] = 1.0  # Default value
        if survival_model:
            try:
                survival_data = prepare_survival_data(df.copy())
                # Predict survival probabilities at multiple times (including 12 months)
                survival_preds = survival_model.predict_survival_function(survival_data, times=[3, 6, 12, 24])
                df['12mo_survival'] = survival_preds.loc[12].values  # Extract just the 12-month probabilities
            except Exception as e:
                app.logger.error(f"12-month survival prediction error: {str(e)}")
                # Fallback to the simple approximation if model fails
                df['12mo_survival'] = np.clip(1 - (df['churn_probability'] * (12/72)), 0, 1)
        else:
           # Use approximation if no survival model available
           df['12mo_survival'] = np.clip(1 - (df['churn_probability'] * (12/72)), 0, 1)
        
        # Generate better targeted recommendations
        def generate_recommendations(row):
            recs = []
            prob = row['churn_probability']
            critical = row['critical_month']
            contract = row['Contract']
            internet = row['InternetService']
            payment = row['PaymentMethod']
            
            # High Risk Actions
            if prob > 0.7:
                recs.append({
                    "icon": "phone",
                    "text": "Immediate retention call with manager",
                    "priority": 1,
                    "type": "Critical"
                })
                if critical < 3:
                    recs.append({
                        "icon": "exclamation-triangle",
                        "text": f"Emergency offer (churn in {critical:.1f} months)",
                        "priority": 1,
                        "type": "Urgent"
                    })
                recs.append({
                    "icon": "percent",
                    "text": "30% discount for 12 months",
                    "priority": 1,
                    "type": "Discount"
                })
            
            # Medium Risk Actions
            elif prob > 0.4:
                recs.append({
                    "icon": "envelope",
                    "text": "Personalized retention email",
                    "priority": 2,
                    "type": "Communication"
                })
                if critical < 6:
                    recs.append({
                        "icon": "clock",
                        "text": f"Priority intervention (churn in {critical:.1f} months)",
                        "priority": 2,
                        "type": "Time-sensitive"
                    })
                recs.append({
                    "icon": "gift",
                    "text": "Loyalty bonus offer",
                    "priority": 2,
                    "type": "Incentive"
                })
            
            # Contract-specific offers
            if contract == 'Month-to-month':
                recs.append({
                    "icon": "file-contract",
                    "text": "15% discount for 1-year contract",
                    "priority": 2,
                    "type": "Contract"
                })
            elif contract == 'One year' and prob > 0.5:
                recs.append({
                    "icon": "file-signature",
                    "text": "Upgrade to 2-year contract with bonus",
                    "priority": 2,
                    "type": "Contract"
                })
            
            # Service-specific offers
            if internet == 'Fiber optic':
                recs.append({
                    "icon": "bolt",
                    "text": "Free speed upgrade for 3 months",
                    "priority": 3,
                    "type": "Service"
                })
            elif internet == 'DSL' and prob > 0.4:
                recs.append({
                    "icon": "wifi",
                    "text": "Upgrade to Fiber at current rate",
                    "priority": 2,
                    "type": "Service"
                })
            
            # Payment-specific offers
            if payment == 'Electronic check':
                recs.append({
                    "icon": "credit-card",
                    "text": "$5 discount for auto-pay setup",
                    "priority": 3,
                    "type": "Payment"
                })
            
            # Sort by priority
            recs.sort(key=lambda x: x['priority'])
            return recs
        
        df['recommendations'] = df.apply(generate_recommendations, axis=1)
        
        # Apply server-side filters
        search = request.args.get('search', '').lower()
        risk_filter = request.args.get('risk', '')
        contract_filter = request.args.get('contract', '')

        if search:
            df = df[df.apply(lambda row: 
                search in str(row['customerID']).lower() or 
                search in str(row.get('Contract', '')).lower() or
                search in str(row.get('InternetService', '')).lower(),
                axis=1
            )]

        if risk_filter:
            df = df[df['risk_level'] == risk_filter]

        if contract_filter:
            df = df[df['Contract'] == contract_filter]
        
        # Pagination
        page = request.args.get('page', 1, type=int)
        per_page = 10
        total_pages = (len(df) // per_page) + (1 if len(df) % per_page else 0)
        
        # Sort and paginate
        df_sorted = df.sort_values(['risk_level', 'churn_probability'], ascending=[False, False])
        paginated_df = df_sorted.iloc[(page-1)*per_page : page*per_page]
        
        return render_template('recommendations.html',
                           customers=paginated_df.to_dict('records'),
                           current_page=page,
                           total_pages=total_pages,
                           search=search,
                           risk_filter=risk_filter,
                           contract_filter=contract_filter,
                           active_page='recommendations')
    
    except Exception as e:
        flash(f'Error: {str(e)}', 'error')
        app.logger.error(f"Recommendations error: {traceback.format_exc()}")
        return redirect(url_for('overview'))



if __name__ == '__main__':
    app.run(debug=True, port=5000)