"""Create a sample PDF about Data Science."""
from pypdf import PdfWriter
from io import BytesIO

# Create PDF content as text
content = """
DATA SCIENCE: THE INTERDISCIPLINARY FIELD

Introduction to Data Science

Data science is an interdisciplinary field that uses scientific methods, processes,
algorithms, and systems to extract knowledge and insights from structured and
unstructured data. It combines aspects of statistics, computer science, and
domain expertise to analyze and interpret complex data.

Key Components of Data Science

1. Statistics and Mathematics
   - Probability theory
   - Statistical inference
   - Hypothesis testing
   - Linear algebra
   - Calculus
   - Optimization

2. Computer Science and Programming
   - Python and R programming
   - SQL and database management
   - Data structures and algorithms
   - Software engineering principles
   - Version control (Git)

3. Domain Knowledge
   - Understanding the business context
   - Industry-specific expertise
   - Problem formulation
   - Result interpretation

The Data Science Workflow

Step 1: Problem Definition
Clearly define the business problem and success metrics. Understand stakeholder
requirements and constraints.

Step 2: Data Collection
Gather data from various sources:
- Databases and data warehouses
- APIs and web scraping
- Surveys and experiments
- Public datasets
- IoT sensors and devices

Step 3: Data Cleaning and Preprocessing
- Handle missing values
- Remove duplicates
- Fix inconsistencies
- Convert data types
- Deal with outliers
- Normalize and standardize features

Step 4: Exploratory Data Analysis (EDA)
- Visualize data distributions
- Identify patterns and trends
- Discover correlations
- Generate hypotheses
- Understand data quality

Step 5: Feature Engineering
- Create new features from existing ones
- Select relevant features
- Transform variables
- Encode categorical variables
- Handle temporal features

Step 6: Model Building
- Select appropriate algorithms
- Train multiple models
- Tune hyperparameters
- Validate performance
- Compare models

Step 7: Model Evaluation
- Accuracy, precision, recall, F1-score
- ROC-AUC for classification
- RMSE, MAE for regression
- Cross-validation
- A/B testing

Step 8: Deployment and Monitoring
- Deploy model to production
- Create APIs or dashboards
- Monitor performance
- Retrain as needed
- Document processes

Essential Tools and Technologies

Programming Languages:
- Python: Most popular for data science
- R: Statistical analysis and visualization
- SQL: Database querying
- Scala/Java: Big data processing

Data Manipulation:
- pandas: Data manipulation in Python
- NumPy: Numerical computing
- dplyr: Data manipulation in R

Visualization:
- Matplotlib and Seaborn (Python)
- Plotly: Interactive visualizations
- Tableau: Business intelligence
- Power BI: Microsoft's BI tool
- ggplot2 (R)

Big Data:
- Apache Spark: Distributed computing
- Hadoop: Distributed storage and processing
- Kafka: Stream processing
- Dask: Parallel computing in Python

Cloud Platforms:
- AWS (Amazon Web Services)
- Google Cloud Platform
- Microsoft Azure
- Databricks

Career Paths in Data Science

Data Analyst
- Focus on data visualization and reporting
- SQL expertise
- Dashboard creation
- Basic statistical analysis

Data Scientist
- Advanced statistical modeling
- Machine learning
- Experimental design
- Feature engineering

Machine Learning Engineer
- Model deployment and scaling
- MLOps practices
- Production systems
- Software engineering skills

Data Engineer
- Data pipeline development
- Database architecture
- ETL processes
- Big data technologies

Business Analyst
- Business intelligence
- Stakeholder communication
- Requirements gathering
- ROI analysis

Best Practices

1. Reproducibility
   - Use version control
   - Document code and processes
   - Create reproducible environments
   - Share notebooks and results

2. Communication
   - Visualize findings effectively
   - Tell data stories
   - Tailor message to audience
   - Present actionable insights

3. Ethics and Privacy
   - Respect data privacy
   - Ensure fairness and avoid bias
   - Be transparent about limitations
   - Follow regulations (GDPR, CCPA)

4. Continuous Learning
   - Stay updated with new tools
   - Read research papers
   - Participate in competitions (Kaggle)
   - Join communities and forums

Industry Applications

Healthcare:
- Disease prediction and diagnosis
- Drug discovery
- Patient monitoring
- Healthcare cost optimization

Finance:
- Fraud detection
- Credit risk assessment
- Algorithmic trading
- Customer churn prediction

Retail:
- Demand forecasting
- Customer segmentation
- Price optimization
- Inventory management

Marketing:
- Customer lifetime value
- Campaign optimization
- Sentiment analysis
- Recommendation engines

Transportation:
- Route optimization
- Demand prediction
- Autonomous vehicles
- Predictive maintenance

Emerging Trends

1. AutoML: Automated machine learning for faster model development
2. Explainable AI: Making models more interpretable
3. Edge Analytics: Processing data at the source
4. DataOps: Agile practices for data analytics
5. Synthetic Data: Generating artificial training data
6. Real-time Analytics: Processing streaming data
7. Ethical AI: Focus on fairness and accountability

Conclusion

Data science is a rapidly evolving field that combines technical skills with
business acumen. Success requires continuous learning, strong communication
skills, and the ability to translate data into actionable insights. As
organizations become more data-driven, the demand for skilled data scientists
continues to grow.
"""

# For simplicity, write as text file first then note it should be PDF
# Since pypdf doesn't easily create PDFs from scratch, we'll note this limitation
with open('c:/Users/Mian/Desktop/RAG Assignment/data/data_science.txt', 'w', encoding='utf-8') as f:
    f.write(content)

print("Created data_science.txt (Note: PDF creation from scratch requires additional libraries like reportlab)")
print("For this demo, we'll use a TXT file with rich content instead.")
