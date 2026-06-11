"""Kira's local voice runtime, split from the former monolithic bot/local_voice.py.

bot/local_voice.py is now a thin compatibility facade that re-exports this
package's public surface. Import from there for the stable public API; the
submodules here are internal structure.

Layering (imports only ever point downward):
    models, util, session, windows_focus      (leaves)
    routing, parsing, executor, webcam_intent  (depend on leaves)
    brain_fallback, desktop_agent, multistep
    dispatcher                                 (the transcript router)
    tts, capture, triggers, runtime            (the I/O + loop layer)
"""
