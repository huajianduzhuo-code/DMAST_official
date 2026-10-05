"""
LLM Call Interceptor for Trajectory Logging
Monkey-patches chat_llm methods to capture exact messages sent to models
"""
import logging
from typing import Dict, List, Any, Callable, Optional
from functools import wraps

class ChatLLMWrapper:
    """Wrapper that preserves original chat_llm object interface while adding logging"""
    
    def __init__(self, original_chat_llm, trajectory_logger, agent_type):
        self.original_chat_llm = original_chat_llm
        self.trajectory_logger = trajectory_logger
        self.agent_type = agent_type
    
    def __call__(self, messages: List[Dict], **kwargs):
        """Main call method with logging"""
        if self.trajectory_logger:
            call_metadata = {
                "kwargs": kwargs,
                "agent_type": self.agent_type,
                "method_name": "__call__"
            }
            
            try:
                response, training_info = self.original_chat_llm(messages, **kwargs)
                
                # Log successful call
                self.trajectory_logger.log_llm_call(
                    agent_type=self.agent_type,
                    messages_sent=messages,
                    response_received=response,
                    call_metadata={**call_metadata, "success": True, "training_info": training_info}
                )
                
                return response, training_info
                
            except Exception as e:
                # Log failed call
                self.trajectory_logger.log_llm_call(
                    agent_type=self.agent_type,
                    messages_sent=messages,
                    response_received=None,
                    call_metadata={**call_metadata, "success": False, "error": str(e)}
                )
                raise e
        else:
            # No logger, just call original method
            return self.original_chat_llm(messages, **kwargs)
    
    def __getattr__(self, name):
        """Delegate all other attribute access to the original object"""
        return getattr(self.original_chat_llm, name)
    
    def __setattr__(self, name, value):
        """Handle attribute setting properly"""
        if name in ['original_chat_llm', 'trajectory_logger', 'agent_type']:
            super().__setattr__(name, value)
        else:
            setattr(self.original_chat_llm, name, value)


class LLMCallInterceptor:
    """Intercepts and logs LLM calls for trajectory analysis"""
    
    def __init__(self, trajectory_logger=None):
        self.trajectory_logger = trajectory_logger
        self.original_methods = {}
    
    def patch_agent_chat_llm(self, agent, agent_type: str):
        """Patch an agent's chat_llm method to log calls"""
        if not hasattr(agent, 'chat_llm'):
            logging.warning(f"Agent {agent_type} has no chat_llm method to patch")
            return
        
        original_method = agent.chat_llm
        agent_key = f"{agent_type}_{id(agent)}"
        self.original_methods[agent_key] = original_method
        
        # Create wrapper that preserves all original functionality
        wrapper = ChatLLMWrapper(original_method, self.trajectory_logger, agent_type)
        
        # Replace the method with the wrapper
        agent.chat_llm = wrapper
        logging.debug(f"✅ Patched {agent_type} chat_llm method for LLM call logging")
    
    def unpatch_agent_chat_llm(self, agent, agent_type: str):
        """Restore original chat_llm method"""
        agent_key = f"{agent_type}_{id(agent)}"
        if agent_key in self.original_methods:
            agent.chat_llm = self.original_methods[agent_key]
            del self.original_methods[agent_key]
            logging.debug(f"🔄 Restored original {agent_type} chat_llm method")
    
    def patch_agents(self, web_agent, attacker=None):
        """Patch both agent and attacker (if present)"""
        self.patch_agent_chat_llm(web_agent, "agent")
        if attacker is not None:
            self.patch_agent_chat_llm(attacker, "attacker")
    
    def unpatch_all(self, web_agent, attacker=None):
        """Restore all original methods"""
        self.unpatch_agent_chat_llm(web_agent, "agent")
        if attacker is not None:
            self.unpatch_agent_chat_llm(attacker, "attacker")
        
        # Clear tracking
        self.original_methods.clear()
        logging.debug("🧹 All LLM call patches removed")

def create_interceptor_for_logger(trajectory_logger):
    """Factory function to create interceptor with logger"""
    return LLMCallInterceptor(trajectory_logger)
