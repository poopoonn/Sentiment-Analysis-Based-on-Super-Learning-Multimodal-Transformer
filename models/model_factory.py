

def build_model(args):
   
   
    model_type = getattr(args.model, "model_type", "base")
    
    if model_type == "base":
        
        from .slmt import build_model as build_base_model
        return build_base_model(args)
    
    elif model_type == "cot_text":
        
        from .slmt_cot_text import build_model as build_cot_text_model
        return build_cot_text_model(args)
    
    elif model_type == "csc_audio":
        
        from .slmt_csc_audio import build_model as build_csc_audio_model
        return build_csc_audio_model(args)
    
    elif model_type == "csc_video":
        
        from .slmt_csc_video import build_model as build_csc_video_model
        return build_csc_video_model(args)
    
    elif model_type == "csc_av":
        
        from .slmt_csc_av import build_model as build_csc_av_model
        return build_csc_av_model(args)
    
    elif model_type == "cot":
        
        from .slmt_cot import build_model as build_cot_model
        return build_cot_model(args)
    
    elif model_type == "enhanced":
        
        from .slmt_layer_enhanced import build_model as build_enhanced_model
        return build_enhanced_model(args)
    
    elif model_type == "text_audio_enhanced":
        
        from .slmt_layer_text_audio_enhanced import build_model as build_text_audio_enhanced_model
        return build_text_audio_enhanced_model(args)
    
    else:
        raise ValueError(f": {model_type}") 