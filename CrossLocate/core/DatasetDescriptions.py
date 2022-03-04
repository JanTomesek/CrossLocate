#!/usr/bin/python3
"""
	Dataset Descriptions

	Created by Jan Tomesek on 4.2.2019.
"""

__author__ = "Jan Tomesek"
__email__ = "itomesek@fit.vutbr.cz"
__copyright__ = "Copyright 2019, Jan Tomesek"


datasetRenderV1 = { 'name': 'geoPose3K',
					'database': 'db',
					'queries': 'query_render_color_result',
					'info': 'geoPose3K_render_color' }

datasetFCN8sV1 = { 	'name': 'geoPose3K',
					'database': 'db',
					'queries': 'query_color_result',
					'info': 'geoPose3K_FCN8s_color' }

datasetDeeplabV1 = {	'name': 'geoPose3K',
						'database': 'db',
						'queries': 'query_deeplab_color_result',
						'info': 'geoPose3K_deeplab_color' }

datasetOriginalV1 = {	'name': 'geoPose3K',
						'database': 'db',
						'queries': 'query_original_color_result',
						'info': 'geoPose3K_original_color' }

# structs copied from geoPose3K_render_color!
datasetDepthV1 = { 	'name': 'geoPose3K',
				 	'database': 'db_depth',
				 	'queries': 'query_depth_rescaled',
					'info': 'geoPose3K_depth' }

# structs copied from geoPose3K_render_color!
datasetSilhouettesV1 = {	'name': 'geoPose3K',
					   		'database': 'db_silhouettes',
					   		'queries': 'query_silhouettes_rescaled',
					   		'info': 'geoPose3K_silhouette' }


datasetSegments = {	'name': 'Sparse',
				   	'database': 'database_segments',
				   	'queries': 'query_segments_result',
				   	'meta': 'GeoPose3K_v2_segments' }

datasetSilhouettes = {	'name': 'Sparse',
					  	'database': 'database_silhouettes',
					  	'queries': 'query_silhouettes_result',
					  	'meta': 'GeoPose3K_v2_silhouettes' }

datasetDepth = {	'name': 'Sparse',
					'database': 'database_depth',
					'queries': 'query_depth_result',
					'meta': 'GeoPose3K_v2_depth' }

datasetOriginalToSegments = {	'name': 'Sparse',
				   				'database': 'database_segments',
				   				'queries': 'query_original_result',
				   				'meta': 'GeoPose3K_v2_original_to_segments' }

datasetOriginalToSilhouettes = {	'name': 'Sparse',
				   					'database': 'database_silhouettes',
				   					'queries': 'query_original_result',
				   					'meta': 'GeoPose3K_v2_original_to_silhouettes' }

datasetOriginalToDepth = {		'name': 'Sparse',
				   				'database': 'database_depth',
				   				'queries': 'query_original_result',
				   				'meta': 'GeoPose3K_v2_original_to_depth' }


datasetUniformSegments2000 = {	'name': 'GeoPose3K_uniform',
				   				'database': 'database_segments',
				   				'queries': 'query_segments_result',
				   				'meta': 'GeoPose3K_uniform_segments_2000m' }

datasetUniformSegments500 = {	'name': 'GeoPose3K_uniform',
				   				'database': 'database_segments',
				   				'queries': 'query_segments_result',
				   				'meta': 'GeoPose3K_uniform_segments_500m' }


datasetSegmentsSwiss = {	'name': 'Sparse',
				   			'database': 'database_segments',
				   			'queries': 'query_segments_result',
				   			'meta': 'GeoPose3K_v2_segments_swiss' }

datasetOriginalToSegmentsSwiss = {	'name': 'Sparse',
				   					'database': 'database_segments',
				   					'queries': 'query_original_result',
				   					'meta': 'GeoPose3K_v2_original_to_segments_swiss' }

datasetOriginalToSilhouettesSwiss = { 'name': 'Sparse',
									  'database': 'database_silhouettes',
									  'queries': 'query_original_result',
									  'meta': 'GeoPose3K_v2_original_to_silhouettes_swiss' }

datasetOriginalToDepthSwiss = { 'name': 'Sparse',
								'database': 'database_depth',
								'queries': 'query_original_result',
								'meta': 'GeoPose3K_v2_original_to_depth_swiss' }


datasetSegmentsResolution = {	'name': 'Sparse',
				   				'database': 'database_segments',
				   				'queries': 'query_segments_resolution_result',
				   				'meta': 'GeoPose3K_v2_segments_resolution' }

datasetOriginalToSegmentsResolution = {	'name': 'Sparse',
				   						'database': 'database_segments',
				   						'queries': 'query_original_resolution_result',
				   						'meta': 'GeoPose3K_v2_original_to_segments_resolution' }

datasetOriginalToSegmentsSwissResolution = { 'name': 'Sparse',
											 'database': 'database_segments',
											 'queries': 'query_original_resolution_result',
											 'meta': 'GeoPose3K_v2_original_to_segments_swiss_resolution' }

datasetUniformOriginalToSegments500 = {	'name': 'GeoPose3K_uniform',
				   						'database': 'database_segments',
				   						'queries': 'query_original_result',
				   						'meta': 'GeoPose3K_uniform_original_to_segments_500m' }


datasetAlpsGP3KSegments = { 'name': 'Uniform',
							'database': 'database_segments',
							'queries': 'query_segments_result',
							'meta': 'Alps_gp3k_segments' }

datasetAlpsGP3KOriginalToSegments = { 	'name': 'Uniform',
										'database': 'database_segments',
										'queries': 'query_original_result',
										'meta': 'Alps_gp3k_original_to_segments' }

datasetAlpsPhotosToSegments = { 'name': 'Uniform',
								'database': 'database_segments',
								'queries': 'query_photos_to_segments_result',
								'meta': 'Alps_photos_to_segments' }

datasetAlpsGP3KSegmentsCompact = {	'name': 'Uniform',
									'database': 'database_segments',
									'queries': 'query_segments_result',
									'meta': 'Alps_gp3k_segments_compact' }

datasetAlpsGP3KOriginalToSegmentsCompact = { 	'name': 'Uniform',
												'database': 'database_segments',
												'queries': 'query_original_result',
												'meta': 'Alps_gp3k_original_to_segments_compact' }

datasetAlpsPhotosToSegmentsCompact = {	'name': 'Uniform',
										'database': 'database_segments',
										'queries': 'query_photos_to_segments_result',
										'meta': 'Alps_photos_to_segments_compact' }

datasetAlpsPhotosToSegmentsResolution = { 'name': 'Uniform',
										  'database': 'database_segments',
										  'queries': 'query_photos_to_segments_resolution_result',
										  'meta': 'Alps_photos_to_segments_resolution' }

datasetAlpsPhotosToSegmentsResolutionCompact = { 'name': 'Uniform',
												  'database': 'database_segments',
												  'queries': 'query_photos_to_segments_resolution_result',
												  'meta': 'Alps_photos_to_segments_resolution_compact' }

datasetAlpsPhotosToSilhouettesCompact = { 'name': 'Uniform',
										  'database': 'database_silhouettes',
										  'queries': 'query_photos_to_segments_result',
										  'meta': 'Alps_photos_to_silhouettes_compact' }

datasetAlpsPhotosToDepthCompact = { 'name': 'Uniform',
									'database': 'database_depth',
									'queries': 'query_photos_to_segments_result',
									'meta': 'Alps_photos_to_depth_compact' }

datasetAlpsPhotosToDepthResolutionCompact = { 'name': 'Uniform',
											  'database': 'database_depth',
											  'queries': 'query_photos_to_segments_resolution_result',
											  'meta': 'Alps_photos_to_depth_compact' }

datasetAlpsDatabaseSilhouettes = {	'name': 'Uniform',
									'database': 'database_silhouettes',
									'queries': '',
									'meta': '' }

datasetAlpsDatabaseDepth = {	'name': 'Uniform',
								'database': 'database_depth',
								'queries': '',
								'meta': '' }


datasetAlpsCH1ToSegments = { 'name': 'Uniform',
							 'database': 'database_segments',
							 'queries': 'query_CH1_to_segments_result',
							 'meta': 'Alps_CH1_to_segments' }

datasetAlpsCH1ToSegmentsResolution = { 'name': 'Uniform',
									   'database': 'database_segments',
									   'queries': 'query_CH1_to_segments_resolution_result',
									   'meta': 'Alps_CH1_to_segments_resolution' }

datasetAlpsCH1ToSilhouettes = { 'name': 'Uniform',
								'database': 'database_silhouettes',
								'queries': 'query_CH1_to_segments_result',
								'meta': 'Alps_CH1_to_silhouettes' }

datasetAlpsCH1ToDepth = { 'name': 'Uniform',
						  'database': 'database_depth',
						  'queries': 'query_CH1_to_segments_result',
						  'meta': 'Alps_CH1_to_depth' }

datasetAlpsCH1ToDepthResolution = { 'name': 'Uniform',
									'database': 'database_depth',
									'queries': 'query_CH1_to_segments_resolution_result',
									'meta': 'Alps_CH1_to_depth' }


datasetAlpsCH2ToSegments = { 'name': 'Uniform',
							 'database': 'database_segments',
							 'queries': 'query_CH2_to_segments_result',
							 'meta': 'Alps_CH2_to_segments' }

datasetAlpsCH2ToSegmentsResolution = { 'name': 'Uniform',
									   'database': 'database_segments',
									   'queries': 'query_CH2_to_segments_resolution_result',
									   'meta': 'Alps_CH2_to_segments_resolution' }

datasetAlpsCH2ToSegmentsExt = { 'name': 'Uniform',
								'database': 'database_segments',
								'queries': 'query_CH2_to_segments_result',
								'meta': 'Alps_CH2_to_segments_ext' }

datasetAlpsCH2ToSegmentsExtResolution = { 'name': 'Uniform',
										  'database': 'database_segments',
										  'queries': 'query_CH2_to_segments_resolution_result',
										  'meta': 'Alps_CH2_to_segments_ext_resolution' }

datasetAlpsCH2ToSilhouettes = { 'name': 'Uniform',
								'database': 'database_silhouettes',
								'queries': 'query_CH2_to_segments_result',
								'meta': 'Alps_CH2_to_silhouettes' }

datasetAlpsCH2ToSilhouettesExt = { 'name': 'Uniform',
								   'database': 'database_silhouettes',
								   'queries': 'query_CH2_to_segments_result',
								   'meta': 'Alps_CH2_to_silhouettes_ext' }

datasetAlpsCH2ToDepth = { 'name': 'Uniform',
						  'database': 'database_depth',
						  'queries': 'query_CH2_to_segments_result',
						  'meta': 'Alps_CH2_to_depth' }

datasetAlpsCH2ToDepthResolution = { 'name': 'Uniform',
									'database': 'database_depth',
									'queries': 'query_CH2_to_segments_resolution_result',
									'meta': 'Alps_CH2_to_depth' }

datasetAlpsCH2ToDepthExt = { 'name': 'Uniform',
							 'database': 'database_depth',
							 'queries': 'query_CH2_to_segments_result',
							 'meta': 'Alps_CH2_to_depth_ext' }

datasetAlpsCH2ToDepthExtResolution = { 'name': 'Uniform',
									   'database': 'database_depth',
									   'queries': 'query_CH2_to_segments_resolution_result',
									   'meta': 'Alps_CH2_to_depth_ext' }


datasetAustriaQ1 = { 'name': 'austria_segments_quarter1',
                     'database': 'db_result',
                     'queries': 'query_result',
                     'info': 'austria_segments_quarter1' }
