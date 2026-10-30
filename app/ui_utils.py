"""UI utility functions, styling tokens, and navigation helpers for HostelWatt."""

import streamlit as st
import plotly.graph_objects as go


def apply_custom_css():
    """Apply modern visual theme, card styles, and KPI styling."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        .metric-card {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.8));
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            padding: 1.25rem 1.5rem;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
            backdrop-filter: blur(12px);
            transition: transform 0.2s ease, border-color 0.2s ease;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            border-color: rgba(99, 102, 241, 0.4);
        }
        .metric-card.card-warning { border-top: 3px solid #f59e0b; }
        .metric-card.card-critical { border-top: 3px solid #ef4444; }
        .metric-card.card-primary { border-top: 3px solid #38bdf8; }
        .metric-card.card-normal { border-top: 3px solid rgba(99, 102, 241, 0.4); }

        .metric-title {
            color: #94a3b8;
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.35rem;
        }

        .metric-value {
            color: #f8fafc;
            font-size: 1.8rem;
            font-weight: 800;
            line-height: 1.15;
            word-break: break-word;
        }

        .metric-sub {
            color: #64748b;
            font-size: 0.8rem;
            margin-top: 0.35rem;
        }

        .status-badge {
            display: inline-block;
            padding: 0.25rem 0.65rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .status-normal { background: rgba(34, 197, 94, 0.15); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.3); }
        .status-warning { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }
        .status-critical { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }

        .disclaimer-banner {
            background: rgba(59, 130, 246, 0.1);
            border-left: 4px solid #3b82f6;
            padding: 0.75rem 1rem;
            border-radius: 6px;
            margin: 1rem 0;
            font-size: 0.85rem;
            color: #93c5fd;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(title: str, subtitle: str, icon: str = "⚡"):
    """Render standardized page header."""
    apply_custom_css()
    st.markdown(
        f"""
        <div style="margin-bottom: 1.75rem; display: flex; align-items: flex-start; gap: 1rem;">
            <div style="font-size: 2.2rem; background: rgba(99, 102, 241, 0.15); padding: 0.4rem 0.8rem; border-radius: 12px; border: 1px solid rgba(99, 102, 241, 0.3);">
                {icon}
            </div>
            <div>
                <h1 style="margin: 0; font-size: 2.1rem; font-weight: 800; background: linear-gradient(90deg, #60a5fa, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    {title}
                </h1>
                <p style="margin: 0.2rem 0 0 0; color: #94a3b8; font-size: 0.95rem;">
                    {subtitle}
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(title: str, value: str, sub: str = "", status: str = "normal"):
    """Render a styled metric tile."""
    status_class = f"card-{status}" if status in ["warning", "critical", "primary", "normal"] else ""
    st.markdown(
        f"""
        <div class="metric-card {status_class}">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            {f'<div class="metric-sub">{sub}</div>' if sub else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_plotly_theme() -> dict:
    """Return consistent, clean dark plotly template styling dictionary."""
    theme_dict = {
        "paper_bgcolor": "rgba(15, 23, 42, 0.0)",
        "plot_bgcolor": "rgba(15, 23, 42, 0.4)",
        "font": {"family": "Plus Jakarta Sans, sans-serif", "color": "#94a3b8"},
        "hoverlabel": {"bgcolor": "#1e293b", "font": {"family": "Plus Jakarta Sans", "color": "#f8fafc"}},
        "xaxis": {"gridcolor": "rgba(255, 255, 255, 0.06)", "zerolinecolor": "rgba(255, 255, 255, 0.1)"},
        "yaxis": {"gridcolor": "rgba(255, 255, 255, 0.06)", "zerolinecolor": "rgba(255, 255, 255, 0.1)"},
        "margin": {"l": 40, "r": 20, "t": 35, "b": 40},
        "legend": {"bgcolor": "rgba(0,0,0,0)", "font": {"color": "#cbd5e1"}},
    }
    # Allow both direct unpacking and theme["layout"] unpacking
    res = dict(theme_dict)
    res["layout"] = theme_dict
    return res


