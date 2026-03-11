import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from pdfminer.high_level import extract_text
from docx import Document
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

# ======================================
# RESUME FRAUD DETECTION - COMPLETE PIPELINE WITH VISUALIZATIONS
# ======================================

class ResumeFraudDetector:
    def __init__(self):
        self.model = None
        self.embedding_model = None
        self.encoder = LabelEncoder()
        self.feature_columns = None
        self.is_trained = False
        self.y_test = None
        self.y_pred_proba = None

    def generate_dataset(self, samples=1000, filename="resume_fraud_dataset.xlsx"):
        """Generate synthetic training dataset"""
        np.random.seed(42)
        data = {
            "experience_years": np.random.randint(0, 15, samples),
            "job_count": np.random.randint(1, 8, samples),
            "average_job_duration": np.random.uniform(0.2, 5, samples),
            "skill_count": np.random.randint(3, 20, samples),
            "education_level": np.random.choice(["Bachelor", "Master", "PhD"], samples),
            "career_gap_months": np.random.randint(0, 24, samples),
            "promotion_rate": np.random.uniform(0, 1.5, samples),
            "salary_growth": np.random.uniform(0, 2, samples),
            "skill_role_match_score": np.random.uniform(0, 1, samples),
            "title_experience_ratio": np.random.uniform(0.1, 4, samples)
        }
        df = pd.DataFrame(data)

        # Fraud detection logic
        df["fraud_label"] = (
            (df["title_experience_ratio"] > 2.5) |
            (df["career_gap_months"] > 12) |
            (df["skill_role_match_score"] < 0.3)
        ).astype(int)

        df.to_excel(filename, index=False)
        print(f"Dataset generated and saved: {filename}")

        # Plot dataset distribution
        self.plot_dataset_distribution(df)

        return df

    def plot_dataset_distribution(self, df):
        """Plot dataset distribution and fraud patterns"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('📊 Resume Fraud Dataset Analysis', fontsize=16, fontweight='bold')

        # 1. Fraud distribution
        df['fraud_label'].value_counts().plot(kind='pie', ax=axes[0,0], autopct='%1.1f%%',
                                            colors=['#4CAF50', '#F44336'], textprops={'fontsize':12})
        axes[0,0].set_title('Fraud vs Legitimate Distribution')
        axes[0,0].set_ylabel('')

        # 2. Key feature distributions
        features = ['title_experience_ratio', 'career_gap_months', 'skill_role_match_score']
        for i, feature in enumerate(features):
            row, col = divmod(i+1, 3)
            df.boxplot(column=feature, by='fraud_label', ax=axes[row-1, col],
                      patch_artist=True, fontsize=10)
            axes[row-1, col].set_title(f'{feature.title()} by Fraud Status')

        plt.tight_layout()
        plt.show()

    def extract_text_from_pdf(self, file_path):
        """Extract text from PDF resume"""
        return extract_text(file_path)

    def extract_text_from_docx(self, file_path):
        """Extract text from DOCX resume"""
        doc = Document(file_path)
        text = [p.text for p in doc.paragraphs]
        return "\n".join(text)

    def parse_resume(self, text):
        """Extract structured features from resume text"""
        skills = re.findall(r'Python|SQL|Machine Learning|Java|C\+\+|Deep Learning', text, re.IGNORECASE)
        experience = re.findall(r'(\d+)\s+years?', text, re.IGNORECASE)
        education = re.findall(r'Bachelor|Master|PhD', text, re.IGNORECASE)

        features = {
            "skill_count": len(skills),
            "experience_years": sum([int(x) for x in experience]) if experience else 0,
            "education_level": self.encoder.transform([education[0]])[0] if self.is_trained and education else 0,
            "job_count": 3,  # default
            "career_gap_months": 2,  # default
            "promotion_rate": 0.3,  # default
            "salary_growth": 0.5,  # default
            "skill_role_match_score": 0.7,  # default
            "title_experience_ratio": 0.5,  # default
            "average_job_duration": 2.0  # default
        }
        return features

    def train(self, dataset_path="resume_fraud_dataset.xlsx"):
        """Train the complete fraud detection model"""
        print("Loading dataset...")
        df = pd.read_excel(dataset_path)

        # Encode categorical features
        df["education_encoded"] = self.encoder.fit_transform(df["education_level"])
        structured_features = df.drop(["fraud_label", "education_level"], axis=1)
        structured_features = pd.get_dummies(structured_features, dummy_na=True)

        # Generate embeddings
        print("Generating text embeddings...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        resume_texts = ["Resume with skills and experience"] * len(df)
        embeddings = self.embedding_model.encode(resume_texts)
        embeddings_df = pd.DataFrame(embeddings)

        # Combine features
        X = pd.concat([structured_features.reset_index(drop=True), embeddings_df], axis=1)
        X.columns = X.columns.astype(str)
        self.feature_columns = X.columns.tolist()
        y = df["fraud_label"]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self.y_test = y_test

        # Train Random Forest
        print("Training Random Forest model...")
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=3,
            random_state=42
        )
        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        self.y_pred_proba = y_pred_proba

        print("\nModel Performance:")
        print(classification_report(y_test, y_pred))

        # Generate comprehensive visualizations
        self.generate_comprehensive_plots(X.columns, y_test, y_pred, y_pred_proba)

        self.is_trained = True
        print("✅ Model training completed!")

    def generate_comprehensive_plots(self, feature_names, y_test, y_pred, y_pred_proba):
        """Generate comprehensive model evaluation plots"""
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Confusion Matrix
        ax1 = fig.add_subplot(gs[0, 0])
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   xticklabels=['Legitimate', 'Fraud'],
                   yticklabels=['Legitimate', 'Fraud'])
        ax1.set_title('Confusion Matrix\n[Accuracy: {:.1%}]'.format((cm[0,0]+cm[1,1])/(cm.sum())))

        # 2. ROC Curve
        ax2 = fig.add_subplot(gs[0, 1])
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curve')
        ax2.legend(loc="lower right")
        ax2.grid(True, alpha=0.3)

        # 3. Feature Importance
        ax3 = fig.add_subplot(gs[0, 2])
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": self.model.feature_importances_
        }).sort_values(by="importance", ascending=True).tail(10)

        colors = plt.cm.viridis(np.linspace(0.3, 1, len(importance_df)))
        ax3.barh(range(len(importance_df)), importance_df['importance'], color=colors)
        ax3.set_yticks(range(len(importance_df)))
        ax3.set_yticklabels(importance_df['feature'], fontsize=9)
        ax3.set_xlabel('Importance Score')
        ax3.set_title('Top 10 Fraud Indicators')

        # 4. Prediction Distribution
        ax4 = fig.add_subplot(gs[1, :2])
        df_pred = pd.DataFrame({
            'Probability': y_pred_proba,
            'Actual': ['Fraud' if x == 1 else 'Legitimate' for x in y_test],
            'Predicted': ['Fraud' if x == 1 else 'Legitimate' for x in y_pred]
        })
        sns.histplot(data=df_pred, x='Probability', hue='Actual', bins=20, ax=ax4,
                    palette=['green', 'red'], alpha=0.7, stat='density')
        ax4.axvline(0.6, color='orange', linestyle='--', label='Decision Threshold (0.6)')
        ax4.set_title('Prediction Probability Distribution')
        ax4.legend()

        # 5. Precision-Recall Curve (placeholder space)
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.text(0.5, 0.5, 'Additional\nMetrics\nComing Soon', ha='center', va='center',
                transform=ax5.transAxes, fontsize=14, fontweight='bold')
        ax5.set_title('Model Metrics')
        ax5.axis('off')

        # 6. Fraud Risk Heatmap
        ax6 = fig.add_subplot(gs[2, :])
        risk_data = pd.DataFrame({
            'title_experience_ratio': np.linspace(0, 4, 10),
            'career_gap_months': np.linspace(0, 24, 10)
        })
        risk_data['fraud_score'] = (
            (risk_data['title_experience_ratio'] > 2.5).astype(int) * 0.5 +
            (risk_data['career_gap_months'] > 12).astype(int) * 0.5
        )
        pivot = risk_data.pivot(index='career_gap_months', columns='title_experience_ratio', values='fraud_score')
        sns.heatmap(pivot, annot=True, cmap='RdYlGn_r', ax=ax6, cbar_kws={'label': 'Fraud Risk Score'})
        ax6.set_title('Fraud Risk Heatmap\n(Title Ratio vs Career Gap)')

        plt.suptitle('🎯 RESUME FRAUD DETECTION - MODEL PERFORMANCE DASHBOARD',
                    fontsize=20, fontweight='bold', y=0.95)
        plt.tight_layout()
        plt.show()

    def plot_feature_importance(self, feature_names, importances):
        """Plot top feature importances (legacy method)"""
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values(by="importance", ascending=False)

        plt.figure(figsize=(12, 8))
        plt.barh(importance_df["feature"][:15], importance_df["importance"][:15],
                color=plt.cm.Purples(np.linspace(0.3, 1, 15)))
        plt.title("🔍 Top Resume Fraud Indicators", fontsize=16, fontweight='bold')
        plt.xlabel("Importance Score", fontsize=12)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

    def predict_resume_fraud(self, resume_text_or_path, is_file_path=False):
        """Predict fraud probability for a resume"""
        if not self.is_trained:
            raise ValueError("Model must be trained first! Call train() method.")

        # Extract text if file path
        if is_file_path:
            if resume_text_or_path.endswith('.pdf'):
                text = self.extract_text_from_pdf(resume_text_or_path)
            elif resume_text_or_path.endswith('.docx'):
                text = self.extract_text_from_docx(resume_text_or_path)
            else:
                text = resume_text_or_path
        else:
            text = resume_text_or_path

        # Parse structured features
        parsed_features = self.parse_resume(text)
        struct_df = pd.DataFrame([parsed_features])

        # Generate embedding
        embedding = self.embedding_model.encode([text])
        emb_df = pd.DataFrame(embedding)

        # Combine features
        features = pd.concat([struct_df, emb_df], axis=1)
        features.columns = features.columns.astype(str)

        # Align with training features
        features = features.reindex(columns=self.feature_columns, fill_value=0)

        # Predict
        probability = self.model.predict_proba(features)[0][1]
        label = "🚨 SUSPICIOUS RESUME" if probability > 0.6 else "✅ LEGITIMATE RESUME"

        self.plot_single_prediction(text[:100] + "...", probability, label)

        return {
            "label": label,
            "fraud_probability": f"{probability:.1%}",
            "confidence": f"{max(self.model.predict_proba(features)[0])*100:.1f}%"
        }

    def plot_single_prediction(self, resume_preview, probability, label):
        """Plot individual prediction visualization"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Gauge chart simulation
        angles = np.linspace(0, 2*np.pi, 100)
        if probability > 0.6:
            color = '#F44336'
            risk_level = 'HIGH RISK'
        else:
            color = '#4CAF50'
            risk_level = 'LOW RISK'

        # Risk gauge
        theta = np.deg2rad(270 + (probability * 180))
        ax1.pie([probability, 1-probability], radius=1.2, colors=[color, 'lightgray'],
                wedgeprops=dict(width=0.4, alpha=0.8))
        ax1.text(0, 0, f'{probability:.0%}\n{risk_level}', ha='center', va='center',
                fontsize=20, fontweight='bold', color=color)
        ax1.set_title(f'🎯 Fraud Risk Score', fontsize=14, fontweight='bold')

        # Feature breakdown (top indicators)
        top_features = ['title_experience_ratio', 'career_gap_months', 'skill_role_match_score']
        values = [0.5, 0.2, 0.7]  # sample values
        colors = ['#FF9800', '#FF5722', '#2196F3']
        bars = ax2.bar(top_features, values, color=colors, alpha=0.8)
        ax2.set_title('📈 Key Fraud Indicators', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Normalized Score')

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=11)

        plt.suptitle(f'🔍 Resume Analysis: {label}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

# ======================================
# USAGE EXAMPLE
# ======================================

if __name__ == "__main__":
    # Initialize detector
    detector = ResumeFraudDetector()

    # Step 1: Generate training data (one-time) - NOW WITH VISUALIZATIONS!
    detector.generate_dataset()

    # Step 2: Train model - NOW WITH COMPREHENSIVE DASHBOARD!
    detector.train()

    # Step 3: Test with sample resumes - NOW WITH INDIVIDUAL PLOTS!
    test_resumes = [
        "Senior AI Engineer with 12 years Python ML Deep Learning experience, PhD Computer Science",
        "Fresh graduate with 15+ years C++ experience, multiple promotions, Senior Architect role",
        "Data Scientist with SQL Python 5 years experience, Master degree, career gap 3 months"
    ]

    print("\n" + "="*60)
    print("🎨 RESUME FRAUD ANALYSIS WITH VISUALIZATIONS")
    print("="*60)

    for i, resume in enumerate(test_resumes, 1):
        print(f"\nAnalyzing Resume {i}...")
        result = detector.predict_resume_fraud(resume)
        print(f"  {result['label']}")
        print(f"  Fraud Risk: {result['fraud_probability']}")
        print(f"  Confidence: {result['confidence']}")
        print("-" * 40)

!git commit -m "Initial commit"
