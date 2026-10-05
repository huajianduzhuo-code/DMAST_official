"""
Subprocess Environment Wrapper
==============================

Provides a subprocess-based environment wrapper with timeout protection
and HTML injection support for adversarial attacks.
"""
import multiprocessing as mp
import logging


class TimeoutException(Exception):
    pass


def _generate_injection_script(injection_dict):
    """
    Generate JavaScript code to execute the HTML injection.
    
    Args:
        injection_dict: Dictionary containing target, position, html, css
        
    Returns:
        JavaScript code string to execute via Selenium
    """
    if not injection_dict or not injection_dict.get('html'):
        return ""
    
    target = injection_dict.get('target', '#area')
    position = injection_dict.get('position', 'prepend')
    html = injection_dict.get('html', '')
    css = injection_dict.get('css', '')
    
    # Escape for JavaScript string
    html_escaped = html.replace('\\', '\\\\').replace('`', '\\`').replace('$', '\\$')
    css_escaped = css.replace('\\', '\\\\').replace('`', '\\`').replace('$', '\\$') if css else ''
    
    script = f"""
(function() {{
    try {{
        // Inject CSS if provided
        if (`{css_escaped}`.trim()) {{
            var style = document.createElement('style');
            style.setAttribute('data-attacker', 'true');
            style.textContent = `{css_escaped}`;
            document.head.appendChild(style);
        }}
        
        // Find target element
        var target = document.querySelector(`{target}`);
        if (!target) {{
            console.error('Injection target not found:', `{target}`);
            return {{ success: false, error: 'Target not found' }};
        }}
        
        // Create wrapper for injected content
        var wrapper = document.createElement('div');
        wrapper.setAttribute('data-attacker', 'true');
        wrapper.innerHTML = `{html_escaped}`;
        
        // Insert based on position
        var position = `{position}`.toLowerCase();
        if (position === 'prepend') {{
            target.insertBefore(wrapper, target.firstChild);
        }} else if (position === 'append') {{
            target.appendChild(wrapper);
        }} else if (position === 'before') {{
            target.parentNode.insertBefore(wrapper, target);
        }} else if (position === 'after') {{
            target.parentNode.insertBefore(wrapper, target.nextSibling);
        }} else {{
            // Default to prepend
            target.insertBefore(wrapper, target.firstChild);
        }}
        
        console.log('HTML injection successful');
        return {{ success: true }};
        
    }} catch (err) {{
        console.error('HTML injection failed:', err);
        return {{ success: false, error: err.message }};
    }}
}})();
"""
    return script


def _step_worker(conn, env_args, action_mapping):
    """
    Worker that creates its own environment and steps it.
    make_env_fn: a function or callable that returns a new env instance
    conn: multiprocessing connection to parent process
    """
    env = env_args.make_env(
        action_mapping=action_mapping,
        exp_dir=None,
    )

    while True:
        try:
            cmd, data = conn.recv()
            if cmd == 'step':
                try:
                    obs, reward, terminated, truncated, info = env.step(data)
                    conn.send(('result', (obs, reward, terminated, truncated, info)))
                except Exception as e:
                    conn.send(('error', str(e)))
                    
            elif cmd == 'reset':
                try:
                    obs, info = env.reset(seed=data)
                    conn.send(('reset_result', (obs, info)))
                except Exception as e:
                    conn.send(('error', str(e)))
                    
            elif cmd == 'inject_html':
                # HTML injection command
                # data is a dict with: injection_dict
                try:
                    injection_dict = data
                    
                    if injection_dict and injection_dict.get('html'):
                        # Get the browser driver/page from the environment
                        # BrowserGym environments may use Selenium (driver) or Playwright (page)
                        driver = None
                        page = None
                        is_playwright = False
                        
                        # Try to find the driver/page in various locations
                        if hasattr(env, 'unwrapped'):
                            unwrapped = env.unwrapped
                            # Check for Playwright page first
                            if hasattr(unwrapped, 'page'):
                                page = unwrapped.page
                                is_playwright = True
                            # Check for Selenium driver
                            elif hasattr(unwrapped, 'driver'):
                                driver = unwrapped.driver
                            elif hasattr(unwrapped, '_driver'):
                                driver = unwrapped._driver
                            elif hasattr(unwrapped, 'instance') and hasattr(unwrapped.instance, 'driver'):
                                # MiniWoB-style environments with Selenium
                                driver = unwrapped.instance.driver
                        
                        if driver is None and page is None:
                            # Try direct access
                            if hasattr(env, 'page'):
                                page = env.page
                                is_playwright = True
                            elif hasattr(env, 'driver'):
                                driver = env.driver
                            elif hasattr(env, '_driver'):
                                driver = env._driver
                        
                        if driver is None and page is None:
                            conn.send(('inject_result', {
                                'success': False, 
                                'error': 'Could not find browser driver/page'
                            }))
                            continue
                        
                        # Generate and execute injection script
                        script = _generate_injection_script(injection_dict)
                        if script:
                            if is_playwright and page is not None:
                                # Playwright uses evaluate() instead of execute_script()
                                result = page.evaluate(script)
                            else:
                                # Selenium uses execute_script()
                                result = driver.execute_script(script)
                            conn.send(('inject_result', result or {'success': True}))
                        else:
                            conn.send(('inject_result', {'success': False, 'error': 'Empty script'}))
                    else:
                        conn.send(('inject_result', {'success': False, 'error': 'No HTML content'}))
                        
                except Exception as e:
                    conn.send(('error', f'HTML injection error: {str(e)}'))
                    
            elif cmd == 'get_observation':
                # Get current observation without stepping
                try:
                    obs = None
                    
                    # Try BrowserGym environment first (check unwrapped for _get_obs)
                    unwrapped = env.unwrapped if hasattr(env, 'unwrapped') else env
                    
                    if hasattr(unwrapped, '_get_obs'):
                        # BrowserGym style - has _get_obs method
                        obs = unwrapped._get_obs()
                    elif hasattr(unwrapped, 'instance') and hasattr(unwrapped.instance, 'get_observation'):
                        # MiniWoB environment interface.
                        obs, extra_info = unwrapped.instance.get_observation()
                    elif hasattr(env, '_get_obs'):
                        obs = env._get_obs()
                    elif hasattr(env, 'get_obs'):
                        obs = env.get_obs()
                    
                    if obs is not None:
                        conn.send(('observation_result', obs))
                    else:
                        conn.send(('error', 'Could not get observation from environment'))
                except Exception as e:
                    conn.send(('error', f'Get observation error: {str(e)}'))
                    
            elif cmd == 'execute_js':
                # Execute arbitrary JavaScript (for advanced attacks)
                try:
                    script = data
                    driver = None
                    page = None
                    is_playwright = False
                    
                    # Find driver/page (same logic as inject_html)
                    if hasattr(env, 'unwrapped'):
                        unwrapped = env.unwrapped
                        if hasattr(unwrapped, 'page'):
                            page = unwrapped.page
                            is_playwright = True
                        elif hasattr(unwrapped, 'instance') and hasattr(unwrapped.instance, 'driver'):
                            driver = unwrapped.instance.driver
                        elif hasattr(unwrapped, 'driver'):
                            driver = unwrapped.driver
                    
                    if driver is None and page is None:
                        if hasattr(env, 'page'):
                            page = env.page
                            is_playwright = True
                        elif hasattr(env, 'driver'):
                            driver = env.driver
                    
                    if driver is None and page is None:
                        conn.send(('js_result', {'success': False, 'error': 'No driver/page found'}))
                        continue
                    
                    if is_playwright and page is not None:
                        result = page.evaluate(script)
                    else:
                        result = driver.execute_script(script)
                    conn.send(('js_result', {'success': True, 'result': result}))
                except Exception as e:
                    conn.send(('error', f'JS execution error: {str(e)}'))
                    
            elif cmd == 'close':
                env.close()
                conn.close()
                break
                
        except EOFError:
            break
        except KeyboardInterrupt:
            break


class TimeoutEnvWrapper:
    def __init__(self, env_args, action_mapping, timeout=5):
        """
        make_env_fn: a function to construct a new env instance
        timeout: max seconds to allow for each step
        """
        self.timeout = timeout
        # Use spawn context for browser environments (safer for cross-process communication)
        ctx = mp.get_context("spawn")
        self.parent_conn, self.child_conn = ctx.Pipe()
        self.process = ctx.Process(target=_step_worker, args=(self.child_conn, env_args, action_mapping))
        self.process.start()
        self.child_conn.close()

    def step(self, action):
        self.parent_conn.send(('step', action))
        if self.parent_conn.poll(self.timeout):
            msg, data = self.parent_conn.recv()
            if msg == 'result':
                return data  # (obs, reward, terminated, truncated, info)
            elif msg == 'error':
                raise RuntimeError(f"Step error: {data}")
        raise TimeoutException("Step timed out (no response), action: {action}")

    def reset(self, seed=None):
        self.parent_conn.send(('reset', seed))
        msg, data = self.parent_conn.recv()
        if msg == 'reset_result':
            return data  # (obs, info)
        elif msg == 'error':
            raise RuntimeError(f"Reset error: {data}")

    def inject_html(self, injection_dict, timeout=None):
        """
        Execute HTML injection on the page.
        
        Args:
            injection_dict: Dictionary with target, position, html, css
            timeout: Optional timeout override
            
        Returns:
            dict with 'success' key and optionally 'error'
        """
        timeout = timeout or self.timeout
        self.parent_conn.send(('inject_html', injection_dict))
        if self.parent_conn.poll(timeout):
            msg, data = self.parent_conn.recv()
            if msg == 'inject_result':
                return data
            elif msg == 'error':
                return {'success': False, 'error': data}
        return {'success': False, 'error': 'Injection timed out'}
    
    def get_observation(self, timeout=None):
        """
        Get the current observation without stepping.
        
        Returns:
            Observation dict or None if failed
        """
        timeout = timeout or self.timeout
        self.parent_conn.send(('get_observation', None))
        if self.parent_conn.poll(timeout):
            msg, data = self.parent_conn.recv()
            if msg == 'observation_result':
                return data
            elif msg == 'error':
                logging.warning(f"Failed to get observation: {data}")
                return None
        logging.warning("Get observation timed out")
        return None
    
    def execute_js(self, script, timeout=None):
        """
        Execute arbitrary JavaScript on the page.
        
        Args:
            script: JavaScript code to execute
            timeout: Optional timeout override
            
        Returns:
            dict with 'success', 'result' or 'error'
        """
        timeout = timeout or self.timeout
        self.parent_conn.send(('execute_js', script))
        if self.parent_conn.poll(timeout):
            msg, data = self.parent_conn.recv()
            if msg == 'js_result':
                return data
            elif msg == 'error':
                return {'success': False, 'error': data}
        return {'success': False, 'error': 'JS execution timed out'}

    def close(self):
        try:
            self.parent_conn.send(('close', None))
        except:
            pass
        self.process.join(timeout=1)
        if self.process.is_alive():
            self.process.terminate()
