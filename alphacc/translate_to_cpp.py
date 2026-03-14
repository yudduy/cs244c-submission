"""
Automated Python→C++ translation for AlphaCC policies.

Takes an evolved Python policy (evolved_policy(memory) -> RemyAction) and
generates a C++ AlphaCCRat sender class for Remy's sender-runner.

Uses GPT-5.3-codex via the OpenAI Responses API to perform the translation,
then optionally compiles and runs parity tests.
"""

import argparse
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

# ── LLM Translation ─────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""\
You are translating a Python congestion control policy to C++ for Remy's
sender-runner simulator. The output must be a drop-in replacement for the
AlphaCCRat class.

## C++ Target Structure

The C++ class inherits from Rat (rat.hh). You must produce TWO sections:

### SECTION: SHARED_STATE
The struct fields inside SharedState. Example:
```cpp
bool initialized = false;
double util_ema = 1.0;
int cooldown = 0;
```

### SECTION: UPDATE_BODY
The body of update_window_and_intersend(). It has access to:
- `_memory.field(0)` = send_ewma
- `_memory.field(1)` = rec_ewma
- `_memory.field(2)` = rtt_ratio
- `_memory.field(3)` = slow_rec_ewma
- `st` = reference to SharedState (auto& st = *_shared)
- `apply_action(int window_increment, double window_multiple, double intersend)`
- `_the_window` = current congestion window (int)
- Standard C++ math: min, max, sqrt, round, etc. (from <algorithm> and <cmath>)

## Translation Rules

1. Python `hasattr` init block → `if (!st.initialized)` block
2. Python dict state `st["key"]` → C++ struct field `st.key`
3. Python `math.sqrt(x)` → C++ `sqrt(x)`
4. Python `int(round(x))` → C++ `static_cast<int>(round(x))`
5. Python `max(a, min(b, c))` → C++ `max(a, min(b, c))`
6. Python `RemyAction(window_increment=X, window_multiple=Y, intersend=Z)`
   → C++ `apply_action(X, Y, Z); return;`
7. Python string phases → C++ enum class Phase
8. All floating-point values use `double`
9. Guard all divisions against zero: `max(1e-6, denominator)`

## Output Format

Produce EXACTLY two code blocks labeled SHARED_STATE and UPDATE_BODY.
No explanation, no markdown outside the code blocks.

```SHARED_STATE
// struct fields here
```

```UPDATE_BODY
// function body here
```
""")


def translate_policy(python_code: str, model: str = "gpt-5.3-codex") -> dict:
    """Use LLM to translate Python policy to C++ sections."""
    from openai import OpenAI
    client = OpenAI()

    prompt = f"Translate this Python policy to C++:\n\n```python\n{python_code}\n```"

    if "codex" in model:
        # Responses API for codex models
        resp = client.responses.create(
            model=model,
            instructions=SYSTEM_PROMPT,
            input=prompt,
            max_output_tokens=4096,
            temperature=0.2,
        )
        text = resp.output_text
    else:
        # Chat completions for other models
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=4096,
            temperature=0.2,
        )
        text = resp.choices[0].message.content

    # Parse the two sections
    sections = {}
    for section_name in ["SHARED_STATE", "UPDATE_BODY"]:
        pattern = rf"```{section_name}\n(.*?)```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            sections[section_name] = match.group(1).strip()
        else:
            # Try alternative patterns
            pattern2 = rf"```cpp\n// {section_name}\n(.*?)```"
            match2 = re.search(pattern2, text, re.DOTALL)
            if match2:
                sections[section_name] = match2.group(1).strip()
            else:
                raise ValueError(
                    f"Could not find {section_name} section in LLM output.\n"
                    f"Raw output:\n{text[:500]}"
                )

    return sections


# ── C++ Code Generation ─────────────────────────────────────────────

HEADER_TEMPLATE = textwrap.dedent("""\
#ifndef ALPHACCRAT_HH
#define ALPHACCRAT_HH

#include <memory>

#include "rat.hh"

class AlphaCCRat : private Rat {{
private:
  struct SharedState {{
{state_fields}
  }};

  std::shared_ptr<SharedState> _shared;

  static WhiskerTree & get_dummy_whiskers();
  void update_window_and_intersend() override;
  void apply_action( int window_increment, double window_multiple, double intersend );

public:
  AlphaCCRat();
  AlphaCCRat( const AlphaCCRat & other );
  AlphaCCRat & operator=( const AlphaCCRat & other );

  using Rat::reset;
  using Rat::packets_received;
  using Rat::send;
  using Rat::next_event_time;
  using Rat::packets_sent;
  using Rat::state_DNA;

  std::string str() const {{ return "AlphaCC/OpenEvolve sender-runner port (auto-translated)"; }}
}};

#endif
""")

SOURCE_TEMPLATE = textwrap.dedent("""\
#include <algorithm>
#include <cmath>

#include "alphaccrat.hh"

using namespace std;

WhiskerTree & AlphaCCRat::get_dummy_whiskers()
{{
  static WhiskerTree dummy;
  return dummy;
}}

AlphaCCRat::AlphaCCRat()
  : Rat( get_dummy_whiskers() ),
    _shared( make_shared<SharedState>() )
{{}}

AlphaCCRat::AlphaCCRat( const AlphaCCRat & other )
  : Rat( other ),
    _shared( other._shared )
{{
}}

AlphaCCRat & AlphaCCRat::operator=( const AlphaCCRat & other )
{{
  if ( this != &other ) {{
    _memory = other._memory;
    _packets_sent = other._packets_sent;
    _packets_received = other._packets_received;
    _track = other._track;
    _last_send_time = other._last_send_time;
    _the_window = other._the_window;
    _intersend_time = other._intersend_time;
    _flow_id = other._flow_id;
    _largest_ack = other._largest_ack;
    _shared = other._shared;
  }}
  return *this;
}}

void AlphaCCRat::apply_action( int window_increment, double window_multiple, double intersend )
{{
  const int raw_window = static_cast<int>( _the_window * window_multiple + window_increment );
  _the_window = min( max( 0, raw_window ), 1000000 );
  _intersend_time = intersend;
}}

void AlphaCCRat::update_window_and_intersend()
{{
  auto & st = *_shared;

{update_body}
}}
""")


def generate_cpp(sections: dict) -> tuple[str, str]:
    """Generate .hh and .cc files from translated sections."""
    # Indent state fields
    state_lines = sections["SHARED_STATE"].split("\n")
    state_indented = "\n".join(f"    {line}" for line in state_lines)

    # Indent update body
    body_lines = sections["UPDATE_BODY"].split("\n")
    body_indented = "\n".join(f"  {line}" for line in body_lines)

    header = HEADER_TEMPLATE.format(state_fields=state_indented)
    source = SOURCE_TEMPLATE.format(update_body=body_indented)

    return header, source


# ── Build & Verify ───────────────────────────────────────────────────

def write_and_build(header: str, source: str, bremy_dir: Path) -> bool:
    """Write generated C++ files and attempt compilation."""
    hh_path = bremy_dir / "remy" / "src" / "alphaccrat.hh"
    cc_path = bremy_dir / "remy" / "src" / "alphaccrat.cc"

    # Backup originals
    for p in [hh_path, cc_path]:
        backup = p.with_suffix(p.suffix + ".bak")
        if p.exists() and not backup.exists():
            p.rename(backup)

    hh_path.write_text(header)
    cc_path.write_text(source)
    print(f"Wrote {hh_path}")
    print(f"Wrote {cc_path}")

    # Try to compile
    result = subprocess.run(
        ["make", "-j4"],
        cwd=bremy_dir / "remy" / "src",
        capture_output=True,
        text=True,
        timeout=120,
    )

    if result.returncode == 0:
        print("Compilation successful!")
        return True
    else:
        print(f"Compilation failed:\n{result.stderr[-500:]}")
        return False


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Translate evolved Python policy to C++ AlphaCCRat"
    )
    parser.add_argument(
        "policy_path",
        type=Path,
        help="Path to Python policy file (e.g., output_remy_evolve_v6/best_policy.py)",
    )
    parser.add_argument(
        "--bremy-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "bremy",
        help="Path to bremy directory",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.3-codex",
        help="LLM model for translation (default: gpt-5.3-codex)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Write .hh/.cc here instead of overwriting bremy/remy/src/",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Compile after translation",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print generated C++ without writing files",
    )

    args = parser.parse_args()

    # Read Python policy
    python_code = args.policy_path.read_text()
    print(f"Read policy from {args.policy_path} ({len(python_code)} chars)")

    # Translate via LLM
    print(f"Translating with {args.model}...")
    sections = translate_policy(python_code, model=args.model)
    print(f"Got SHARED_STATE ({len(sections['SHARED_STATE'])} chars) "
          f"and UPDATE_BODY ({len(sections['UPDATE_BODY'])} chars)")

    # Generate C++
    header, source = generate_cpp(sections)

    if args.dry_run:
        print("\n=== alphaccrat.hh ===")
        print(header)
        print("\n=== alphaccrat.cc ===")
        print(source)
        return

    # Write files
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "alphaccrat.hh").write_text(header)
        (args.output_dir / "alphaccrat.cc").write_text(source)
        print(f"Wrote to {args.output_dir}/")
    else:
        write_and_build(header, source, args.bremy_dir)

    if args.build and not args.output_dir:
        success = write_and_build(header, source, args.bremy_dir)
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main()
