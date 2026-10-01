# ============================================================
# TAB 2 – HISTORIE SIGNÁLŮ
# ============================================================
with tab_history:
    st.subheader("Uložené signály")
    if sb is None:
        st.info("Historie není dostupná: není nakonfigurováno připojení Supabase.")
    else:
        col_f1, col_f2 = st.columns([1, 1])
        with col_f1:
            limit = st.number_input("Max. řádků", 50, 2000, 500, 50)
        with col_f2:
            only_today = st.checkbox("Pouze dnešní", False)

        if st.button("🔄 Načíst historii", key="load_history"):
            try:
                q = sb.table("scanner_signals").select("*").order("signal_date", desc=True)
                if only_today:
                    today = datetime.now(timezone.utc).date().isoformat()
                    q = q.eq("signal_date", today)
                resp = q.limit(int(limit)).execute()
                hist = pd.DataFrame(resp.data or [])
                st.session_state.history_df = hist
            except Exception as e:
                log.exception("Načtení historie selhalo")
                st.error(f"Historii se nepodařilo načíst: {e}")

        hist = st.session_state.get("history_df")
        if hist is not None:
            if hist.empty:
                st.info("V databázi zatím nejsou uložené signály.")
            else:
                st.dataframe(hist, use_container_width=True, hide_index=True)
                st.download_button(
                    "📥 Stáhnout historii CSV",
                    hist.to_csv(index=False).encode("utf-8-sig"),
                    "klondike_historie.csv", "text/csv"
                )

# ============================================================
# TAB 3 – UČÍCÍ SE PŘEHLED
# ============================================================
with tab_learning:
    st.subheader("Vyhodnocení historických signálů")
    st.caption("Pouze nákupní signály 4.4. Model: příští open v zóně, kontrola R:R, výstup do 5 dnů. "
               "Jde o simulaci bez nákladů, nikoli validaci ziskovosti. Denní OHLC data nemusí určit "
               "pořadí zásahu stopu a cíle v rámci stejného dne.")

    if sb is None:
        st.info("Učící přehled vyžaduje připojení Supabase.")
    else:
        days = st.slider("Historie (dny)", 30, 365, 90, key="learning_days")

        if st.button("🧠 Vyhodnotit signály", key="run_learning"):
            with st.spinner("Vyhodnocuji historické signály…"):
                learning = load_learning_stats(sb, days=days)
            st.session_state.learning_df = learning

        learning = st.session_state.get("learning_df")
        if learning is not None:
            if learning.empty:
                st.info("Nejsou dostupná data k vyhodnocení.")
            else:
                # Statistiky
                counts = learning["outcome"].value_counts()
                tp1 = int(counts.get("TP1", 0))
                sl  = int(counts.get("SL", 0))
                amb = int(counts.get("AMBIGUOUS", 0))
                opn = int(counts.get("TIME_EXIT", 0))
                pend = int(counts.get("pending", 0))
                not_filled = int(counts.get("NOT_FILLED", 0))

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("TP1", tp1)
                c2.metric("SL", sl)
                c3.metric("Ambiguous", amb)
                c4.metric("Výstup 5. den", opn)
                c5.metric("Vstup nevyplněn", not_filled)
                if pend:
                    st.caption(f"Čeká na dokončení 5 obchodních dnů: {pend} signálů.")

                resolved = tp1 + sl
                if resolved > 0:
                    win_rate = tp1 / resolved * 100
                    st.metric("TP1 / (TP1 + SL); bez časových a nejasných výstupů",
                              f"{win_rate:.1f} %",
                              help=f"Vyhodnoceno: {resolved} obchodů")

                # Průměrný forward return podle outcome
                valid = learning.dropna(subset=["forward_return_5d"])
                if not valid.empty:
                    st.markdown("#### Průměrný výnos (výstup nejpozději 5. den) podle výsledku")
                    agg = valid.groupby("outcome")["forward_return_5d"].agg(
                        ["count", "mean"]).round(2)
                    st.dataframe(agg, use_container_width=True)

                # Průměrný return podle signálu
                if not valid.empty:
                    st.markdown("#### Průměrný výnos (výstup nejpozději 5. den) podle typu signálu")
                    agg2 = valid.groupby("signal")["forward_return_5d"].agg(
                        ["count", "mean"]).round(2)
                    st.dataframe(agg2, use_container_width=True)

                st.markdown("#### Detailní data")
                st.dataframe(learning, use_container_width=True, hide_index=True)

                st.download_button(
                    "📥 Stáhnout vyhodnocení CSV",
                    learning.to_csv(index=False).encode("utf-8-sig"),
                    "klondike_learning.csv", "text/csv"
                )
