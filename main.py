import argparse
import os
from src.system.football_analyzer import FootballAnalyzer
from src.config.config import ConfigManager
import logging

def parse_arguments():
    parser = argparse.ArgumentParser(description='Football Match Analysis System')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to the input image')
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output', type=str, default='output',
                       help='Output directory for results')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    return parser.parse_args()

def main():
    # Argümanları parse et
    args = parse_arguments()

    # Logging ayarla
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    try:
        # Konfigürasyonu yükle
        config_manager = ConfigManager(args.config)
        config = config_manager.get_config()
        
        # Debug modu ayarla
        config.debug_mode = args.debug

        # Çıktı dizinini oluştur
        os.makedirs(args.output, exist_ok=True)

        # Analiz sistemini başlat
        analyzer = FootballAnalyzer(config)

        # Görüntüyü işle
        logger.info(f"Processing image: {args.image}")
        frame = analyzer.process_image(args.image)

        if frame:
            # Sonuçları kaydet
            analyzer.save_results(frame, args.output)
            logger.info("Processing completed successfully")
        else:
            logger.error("Failed to process image")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 