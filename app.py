import json

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Telecom Churn Risk Demo", page_icon="📞", layout="centered")


@st.cache_resource
def load():
    bundle = joblib.load("model.joblib")
    return bundle["model"], bundle["threshold"], bundle["features"], json.load(open("metrics.json"))


model, default_threshold, FEATURES, m = load()

st.title("📞 Telecom Churn Risk Demo")
st.warning(
    "**Demo only, not for real retention decisions.** Trained on a sample dataset of "
    f"{m['rows']:,} customers. It ranks customers by how much they resemble past churners."
)

with st.form("customer"):
    c1, c2 = st.columns(2)
    weeks = c1.number_input("Account age (weeks)", 1, 300, 100)
    renew = c2.selectbox("Contract renewed recently?", ["Yes", "No"])
    plan = c1.selectbox("Has a data plan?", ["No", "Yes"])
    data = c2.number_input("Data usage (GB / month)", 0.0, 10.0, 0.0, step=0.1)
    calls = c1.number_input("Customer service calls", 0, 15, 1)
    daymins = c2.number_input("Day minutes / month", 0.0, 500.0, 180.0, step=5.0)
    daycalls = c1.number_input("Day calls / month", 0, 250, 100)
    charge = c2.number_input("Monthly charge", 10.0, 150.0, 55.0, step=1.0)
    overage = c1.number_input("Overage fee", 0.0, 30.0, 10.0, step=0.5)
    roam = c2.number_input("Roaming minutes", 0.0, 30.0, 10.0, step=0.5)
    go = st.form_submit_button("Score this customer")
st.caption("Monthly charge rises with usage in this data, so use realistic combinations.")

threshold = st.slider(
    "Alert threshold (lower = catch more churners, more false alarms)",
    0.05, 0.95, float(default_threshold), 0.01,
)

if go:
    row = pd.DataFrame([{
        "AccountWeeks": weeks, "ContractRenewal": int(renew == "Yes"),
        "DataPlan": int(plan == "Yes"), "DataUsage": data, "CustServCalls": calls,
        "DayMins": daymins, "DayCalls": daycalls, "MonthlyCharge": charge,
        "OverageFee": overage, "RoamMins": roam,
    }])[FEATURES]
    p = float(model.predict_proba(row)[0, 1])
    st.metric("Churn risk score", f"{p:.0%}")
    st.progress(min(max(p, 0.0), 1.0))
    if p >= threshold:
        st.error("At risk: flag for the retention team.")
    else:
        st.success("Below the alert threshold.")

st.subheader("What the data says")
s = m["segments"]
st.write(
    f"- Customers **without** a recent contract renewal churned **{s['no_contract_renewal']:.0%}** "
    f"of the time, vs **{s['contract_renewed']:.0%}** with one.\n"
    f"- Customers with **4+ service calls** churned **{s['4plus_service_calls']:.0%}** of the time, "
    f"vs **{s['under_4_service_calls']:.0%}** with fewer."
)
imp = pd.Series(m["feature_importance"]).sort_values()
st.bar_chart(imp, horizontal=True)
st.caption("Importance = how much the model's accuracy drops when that column is shuffled.")

with st.expander("How good is this model? (honest numbers)"):
    a, b = m["at_0.5"], m["at_chosen_threshold"]
    st.write(
        f"**Model:** {m['selected_model']}. **Expected ROC-AUC: about {m['headline_auc']:.2f}** "
        f"(± {m['headline_auc_std']:.2f}, repeated cross-validation over all {m['rows']:,} customers). "
        f"A single 80/20 split gave {m['test_auc']:.2f}; one split is noisy with only "
        f"~{b['churners_in_test']} churners in the test set."
    )
    st.write(
        f"On that held-out test set of {m['test_rows']} customers ({b['churners_in_test']} real churners):\n"
        f"- At threshold 0.50: caught **{a['churners_caught']}** of {a['churners_in_test']} churners "
        f"(recall {a['recall']:.0%}), {a['false_alarms']} false alarms.\n"
        f"- At threshold {b['threshold']:.2f}: caught **{b['churners_caught']}** of "
        f"{b['churners_in_test']} (recall {b['recall']:.0%}), {b['false_alarms']} false alarms."
    )
    st.caption(
        "The threshold was chosen on training data only, favouring recall because missing a "
        "churner costs more than a false alarm."
    )
