# Titanic Survival Analysis

<div style="background: linear-gradient(135deg, #1f77b4 0%, #2ca02c 100%); color: white; padding: 2em; border-radius: 10px; margin: 2em 0; text-align: center;">
  <h2 style="color: white; margin-top: 0;">🚢 RMS Titanic: A Data Science Case Study</h2>
  <p style="font-size: 1.1em; margin-bottom: 0;">Exploring survival patterns through comprehensive machine learning analysis</p>
</div>

## Project Overview

The RMS Titanic disaster offers a tragic but data-rich case study in survival analysis. On April 15, 1912, the "unsinkable" ship struck an iceberg and sank, resulting in the deaths of over 1,500 passengers and crew. The disaster has been extensively documented, providing a comprehensive dataset that allows us to explore patterns in survival.

This project demonstrates a complete machine learning workflow, from initial data exploration through model development to final interpretation. We'll investigate which factors most strongly predicted survival and compare multiple classification approaches.

## Project Structure

This analysis is organized into four interconnected notebooks:

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.5em; margin: 2em 0;">

<div style="background: #f8f9fa; padding: 1.5em; border-radius: 8px; border-left: 4px solid #1f77b4;">
  <h3 style="margin-top: 0; color: #1f77b4;">📊 1. Data Exploration</h3>
  <p style="margin-bottom: 0.5em;"><strong><a href="01_exploration.ipynb">Explore →</a></strong></p>
  <p style="font-size: 0.9em; color: #666; margin: 0;">Initial profiling, univariate and bivariate analysis, missing data patterns</p>
</div>

<div style="background: #f8f9fa; padding: 1.5em; border-radius: 8px; border-left: 4px solid #ff7f0e;">
  <h3 style="margin-top: 0; color: #ff7f0e;">🧹 2. Data Cleaning</h3>
  <p style="margin-bottom: 0.5em;"><strong><a href="02_cleaning.ipynb">Explore →</a></strong></p>
  <p style="font-size: 0.9em; color: #666; margin: 0;">Handling missing values, outlier detection, feature engineering</p>
</div>

<div style="background: #f8f9fa; padding: 1.5em; border-radius: 8px; border-left: 4px solid #2ca02c;">
  <h3 style="margin-top: 0; color: #2ca02c;">🤖 3. Modeling</h3>
  <p style="margin-bottom: 0.5em;"><strong><a href="03_modeling.ipynb">Explore →</a></strong></p>
  <p style="font-size: 0.9em; color: #666; margin: 0;">Training and comparing multiple classification algorithms</p>
</div>

<div style="background: #f8f9fa; padding: 1.5em; border-radius: 8px; border-left: 4px solid #d62728;">
  <h3 style="margin-top: 0; color: #d62728;">📈 4. Results</h3>
  <p style="margin-bottom: 0.5em;"><strong><a href="04_results.ipynb">Explore →</a></strong></p>
  <p style="font-size: 0.9em; color: #666; margin: 0;">Model comparison, interpretation, and final insights</p>
</div>

</div>

## Key Questions

<div style="background: #f8f9fa; padding: 1.5em; border-radius: 8px; margin: 1.5em 0;">

- 🔍 Which demographic factors were most predictive of survival?
- 🔍 How did class, gender, and age interact in determining survival outcomes?
- 🔍 Which machine learning model best captures the underlying patterns?
- 🔍 What can we learn about the "women and children first" evacuation policy from the data?

</div>

## Dataset

<div style="background: #e8f4f8; padding: 1.5em; border-radius: 8px; margin: 1.5em 0;">

The dataset contains information on **891 passengers**, including:

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1em; margin-top: 1em;">

<div>
  <strong>👥 Demographics</strong><br>
  <span style="font-size: 0.9em; color: #666;">Age, Sex, Embarked port</span>
</div>

<div>
  <strong>💰 Socioeconomic</strong><br>
  <span style="font-size: 0.9em; color: #666;">Passenger class (Pclass), Fare</span>
</div>

<div>
  <strong>👨‍👩‍👧‍👦 Family</strong><br>
  <span style="font-size: 0.9em; color: #666;">Siblings/spouses, Parents/children</span>
</div>

<div>
  <strong>🎯 Target</strong><br>
  <span style="font-size: 0.9em; color: #666;">Survival (0 = No, 1 = Yes)</span>
</div>

</div>

</div>

## Technical Approach

<div style="background: #fff5e6; padding: 1.5em; border-radius: 8px; border-left: 4px solid #ff7f0e; margin: 1.5em 0;">

We'll employ a rigorous methodology:

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1em; margin-top: 1em;">

<div>
  <strong>📊 Exploratory Data Analysis</strong><br>
  <span style="font-size: 0.9em; color: #666;">Automated profiling and custom visualizations</span>
</div>

<div>
  <strong>🔧 Statistical Imputation</strong><br>
  <span style="font-size: 0.9em; color: #666;">Multiple imputation for missing age values</span>
</div>

<div>
  <strong>🤖 Model Comparison</strong><br>
  <span style="font-size: 0.9em; color: #666;">Logistic regression, tree-based methods, ensemble techniques</span>
</div>

<div>
  <strong>🔍 Model Interpretation</strong><br>
  <span style="font-size: 0.9em; color: #666;">Feature importance, SHAP values, partial dependence plots</span>
</div>

<div>
  <strong>✅ Validation</strong><br>
  <span style="font-size: 0.9em; color: #666;">Stratified cross-validation with multiple evaluation metrics</span>
</div>

</div>

</div>

## Expected Outcomes

<div style="background: #e8f5e9; padding: 1.5em; border-radius: 8px; margin: 1.5em 0;">

By the end of this analysis, we'll have:

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1em; margin-top: 1em;">

<div>
  ✅ <strong>Identified</strong> the strongest predictors of survival
</div>

<div>
  ✅ <strong>Trained</strong> and compared multiple classification models
</div>

<div>
  ✅ <strong>Provided</strong> interpretable explanations for model predictions
</div>

<div>
  ✅ <strong>Documented</strong> a reproducible analytical workflow
</div>

</div>

</div>

---

<div style="text-align: center; margin: 2em 0; padding: 1.5em; background: #f0f0f0; border-radius: 8px;">
  <p style="font-size: 1.2em; margin: 0;"><strong>Ready to explore?</strong></p>
  <p style="margin: 0.5em 0 0 0;"><a href="01_exploration.ipynb" style="background: #1f77b4; color: white; padding: 0.7em 2em; text-decoration: none; border-radius: 5px; display: inline-block;">Start with Data Exploration →</a></p>
</div>
