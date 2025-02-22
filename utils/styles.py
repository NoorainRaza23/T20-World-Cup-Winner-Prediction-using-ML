def load_css():
    return """
    <style>
    /* Light theme variables remain unchanged */
    :root {
        --background-color: #ffffff;
        --text-color: #1e293b;
        --heading-color: #0f172a;
        --subheading-color: #1e293b;
        --accent-color: #1e40af;
        --link-color: #1e40af;
        --link-hover-color: #1e3a8a;
        --border-color: #64748b;
        --shadow-color: rgba(0, 0, 0, 0.1);
        --button-bg: #1e40af;
        --button-hover: #1e3a8a;
        --button-text: #ffffff;
        --input-border: #64748b;
        --input-focus: #2563eb;
        --error-color: #b91c1c;
        --success-color: #15803d;
    }

    /* Dark theme variables with sophisticated background colors */
    [data-theme="dark"] {
        --background-color: #0f172a;       /* Rich navy background */
        --background-gradient: linear-gradient(135deg, #0f172a, #1e293b); /* Subtle gradient */
        --card-bg: #1e293b;                /* Lighter navy for cards */
        --card-hover: #334155;             /* Even lighter for hover states */
        --floating-bg: #2d3748;            /* For elevated components */

        --text-color: #ffffff;             /* Pure white text */
        --heading-color: #ffffff;          /* Pure white headings */
        --subheading-color: #f8fafc;       /* Almost white subheadings */

        --accent-color: #818cf8;           /* Bright indigo */
        --accent-color-light: #93c5fd;     /* Light blue for hover */
        --accent-color-dark: #6366f1;      /* Darker accent for active */

        --link-color: #38bdf8;             /* Bright sky blue */
        --link-hover-color: #7dd3fc;       /* Lighter blue hover */

        --border-color: #475569;           /* Subtle border */
        --shadow-color: rgba(0, 0, 0, 0.5); /* Deep shadow */

        --button-bg: linear-gradient(135deg, #818cf8, #6366f1);
        --button-hover: linear-gradient(135deg, #93c5fd, #818cf8);
        --button-text: #ffffff;

        --input-border: #475569;
        --input-focus: #38bdf8;
        --error-color: #f87171;
        --success-color: #4ade80;
    }

    /* Base Styles with Enhanced Background */
    .stApp {
        background: var(--background-gradient) !important;
        color: var(--text-color);
        line-height: 1.6;
        min-height: 100vh;
    }

    /* Card-like Elements with Depth */
    .stMarkdown, 
    .stDataFrame,
    .stMetric,
    .stSelectbox > div,
    .stTextInput > div {
        background: var(--card-bg);
        border-radius: 0.75rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1),
                   0 10px 15px -3px rgba(0, 0, 0, 0.1);
        padding: 1.25rem;
        margin: 0.5rem 0;
        border: 1px solid var(--border-color);
    }

    /* Floating Elements with Extra Depth */
    .stButton > button,
    .stDownloadButton > button {
        background: var(--button-bg);
        color: var(--button-text);
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 0.5rem;
        font-weight: 600;
        transition: all 0.2s ease;
        cursor: pointer;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1),
                   0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }

    /* Sidebar with Distinct Background */
    [data-testid="stSidebar"] {
        background: var(--card-bg) !important;
        border-right: 1px solid var(--border-color);
        box-shadow: 4px 0 8px rgba(0, 0, 0, 0.1);
    }

    /* Typography with Enhanced Visibility */
    h1, h2, h3, h4, h5, h6 {
        color: var(--heading-color);
        margin-bottom: 1.25rem;
        font-weight: 700;
        line-height: 1.3;
        letter-spacing: -0.02em;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }

    p, li, span {
        color: var(--text-color);
        line-height: 1.7;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
    }

    .stButton > button:hover {
        background: var(--button-hover);
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
    }

    .stButton > button:active {
        transform: translateY(1px);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }

    .stTextInput > div > div > input,
    .stSelectbox > div > div,
    .stTextArea > div > div > textarea {
        background: var(--card-bg);
        color: var(--text-color);
        border: 2px solid var(--input-border);
        border-radius: 0.5rem;
        padding: 0.75rem 1rem;
        font-size: 1rem;
        transition: all 0.2s ease;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div:focus-within,
    .stTextArea > div > div > textarea:focus {
        border-color: var(--input-focus);
        box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.4),
                   inset 0 2px 4px rgba(0, 0, 0, 0.1);
        outline: none;
    }

    .stDataFrame table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        border: 2px solid var(--border-color);
        border-radius: 0.5rem;
        overflow: hidden;
        background: var(--card-bg);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }

    .stDataFrame th {
        background: var(--accent-color-dark);
        color: #ffffff;
        font-weight: 600;
        text-align: left;
        padding: 1rem;
        border-bottom: 2px solid var(--border-color);
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
    }

    .stDataFrame td {
        padding: 1rem;
        color: var(--text-color);
        border-bottom: 1px solid var(--border-color);
        transition: background-color 0.2s ease;
    }

    .stDataFrame tr:hover td {
        background: var(--card-hover);
    }

    .success-message {
        color: var(--success-color);
        font-weight: 600;
        padding: 1rem;
        background: rgba(74, 222, 128, 0.1);
        border-radius: 0.5rem;
        border-left: 4px solid var(--success-color);
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    .error-message {
        color: var(--error-color);
        font-weight: 600;
        padding: 1rem;
        background: rgba(248, 113, 113, 0.1);
        border-radius: 0.5rem;
        border-left: 4px solid var(--error-color);
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    a {
        color: var(--link-color);
        text-decoration: none;
        font-weight: 500;
        border-bottom: 2px solid transparent;
        transition: all 0.2s ease;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
    }

    a:hover {
        color: var(--link-hover-color);
        border-bottom-color: var(--link-hover-color);
    }

    .stRadio > label {
        color: var(--text-color) !important;
        font-weight: 500;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
    }

    .stRadio > div[role="radiogroup"] > label {
        background: var(--card-bg);
        border: 2px solid var(--border-color);
        border-radius: 0.5rem;
        padding: 0.75rem 1rem;
        transition: all 0.2s ease;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    .stRadio > div[role="radiogroup"] > label:hover {
        border-color: var(--accent-color);
        background: var(--card-hover);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }

    [data-testid="stSidebar"] {
        background: var(--card-bg) !important;
        border-right: 1px solid var(--border-color);
        padding: 2rem 1rem;
        box-shadow: 4px 0 8px rgba(0, 0, 0, 0.1);
    }

    ::-webkit-scrollbar {
        width: 12px;
        height: 12px;
    }

    ::-webkit-scrollbar-track {
        background: var(--card-bg);
        border-radius: 6px;
    }

    ::-webkit-scrollbar-thumb {
        background: var(--accent-color);
        border-radius: 6px;
        border: 3px solid var(--card-bg);
        transition: background-color 0.2s ease;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--accent-color-light);
    }

    [data-testid="stMetricValue"] {
        color: var(--heading-color);
        font-weight: 700;
        font-size: 1.5rem;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }

    [data-testid="stMetricDelta"] {
        background: rgba(129, 140, 248, 0.1);
        border-radius: 0.5rem;
        padding: 0.5rem 0.75rem;
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    *:focus-visible {
        outline: 3px solid var(--accent-color-light);
        outline-offset: 2px;
        box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.4);
    }
    </style>
    """