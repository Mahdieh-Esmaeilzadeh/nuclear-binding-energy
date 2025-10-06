# -*- coding: utf-8 -*-
# Script: data_preprocessing.py
# Generated automatically from BindingEnergy.ipynb

try:
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset loaded successfully: {df.shape}")
except Exception as e:
    raise SystemExit(f"Exiting due to dataset loading error: {str(e)}")

print("\nHead:")
print(df.head())

print("\nInfo:")
print(df.info())

print("\nMissing values per column (descending):")
print(df.isnull().sum().sort_values(ascending=False))

if TARGET_COLUMN not in df.columns:
    raise ValueError(f"Target column '{TARGET_COLUMN}' not found. Available: {df.columns.tolist()}")

# Drop columns with >50% missing
missing_pct = df.isnull().mean()
cols_to_drop = missing_pct[missing_pct > 0.5].index.tolist()
if cols_to_drop:
    print(f"\nDropping columns with >50% missing: {cols_to_drop}")
    df = df.drop(columns=cols_to_drop)

# Separate features/target
y = df[TARGET_COLUMN]
X = df.drop(columns=[TARGET_COLUMN])

# Identify column types
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
print(f"\nNumeric columns ({len(numeric_cols)}): {numeric_cols[:10]}{'...' if len(numeric_cols)>10 else ''}")
print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols[:10]}{'...' if len(categorical_cols)>10 else ''}")

try:
    y_bins = pd.qcut(y, q=10, duplicates="drop", labels=False)
except Exception:
    # Fallback if qcut fails (e.g., too many duplicates)
    y_bins = pd.cut(y, bins=10, labels=False)

X_train, X_test, y_train, y_test, bins_train, bins_test = train_test_split(
    X, y, y_bins,
    test_size=0.2,
    random_state=42,
    stratify=y_bins
)

print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")

ohe = OneHotEncoder(
    handle_unknown="ignore",
    min_frequency=0.01
)

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("power", PowerTransformer(method="yeo-johnson", standardize=False)),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", ohe)
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ],
    remainder="drop"  # keep only specified columns
)

# Fit preprocessing on TRAIN ONLY to avoid leakage
preprocessor.fit(X_train)

# Transform
X_train_proc = preprocessor.transform(X_train).toarray()
X_test_proc = preprocessor.transform(X_test).toarray()

# Feature names (for reference)
try:
    feature_names = preprocessor.get_feature_names_out().tolist()
except Exception:
    feature_names = [f"f{i}" for i in range(X_train_proc.shape[1])]

print(f"\nProcessed feature matrix: train={X_train_proc.shape}, test={X_test_proc.shape}")

# Save preprocessor and metadata
joblib.dump(preprocessor, os.path.join(ARTIFACT_DIR, "preprocessor.pkl"))
with open(os.path.join(ARTIFACT_DIR, "feature_names.json"), "w") as f:
    json.dump(feature_names, f, indent=2)

def build_model(input_dim: int,
                width: int = 256,
                blocks: int = 3,
                dropout: float = 0.2,
                l2_reg: float = 1e-4,
                lr: float = 1e-3,
                weight_decay: float = 1e-5) -> tf.keras.Model:
    inp = layers.Input(shape=(input_dim,), name="features")

    # Stem
    x = layers.Dense(width, kernel_initializer="he_normal",
                     kernel_regularizer=regularizers.l2(l2_reg))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("gelu")(x)
    x = layers.Dropout(dropout)(x)

    # Residual blocks (keep same width to allow skip connections)
    for i in range(blocks):
        shortcut = x
        y = layers.Dense(width, kernel_initializer="he_normal",
                         kernel_regularizer=regularizers.l2(l2_reg))(x)
        y = layers.BatchNormalization()(y)
        y = layers.Activation("gelu")(y)
        y = layers.Dropout(dropout)(y)

        y = layers.Dense(width, kernel_initializer="he_normal",
                         kernel_regularizer=regularizers.l2(l2_reg))(y)
        y = layers.BatchNormalization()(y)
        # Pre-activation residual connection
        x = layers.Add()([shortcut, y])
        x = layers.Activation("gelu")(x)
        x = layers.Dropout(dropout)(x)

    # Head
    out = layers.Dense(1, name="y_scaled")(x)

    # Optimizer: AdamW if available, else Adam
    try:
        opt = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay)
    except Exception:
        opt = tf.keras.optimizers.Adam(learning_rate=lr)

    model = models.Model(inputs=inp, outputs=out, name="ResidualMLPRegressor")
    model.compile(
        optimizer=opt,
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae"),
                 tf.keras.metrics.RootMeanSquaredError(name="rmse")]
    )
    return model

model = build_model(
    input_dim=X_train_proc.shape[1],
    width=256,
    blocks=3,
    dropout=0.25,
    l2_reg=1e-4,
    lr=1e-3,
    weight_decay=1e-5
)

model.summary()

