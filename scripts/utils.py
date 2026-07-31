#!/usr/bin/env python3


import os
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib


def get_col_names(name: str) -> list[str]:
    return {
        "basic_info_cols": [
            "AGE", 
            "SEX",
            "LivingPlace", 
            "Education",
            "OccupationStatus",
            "EducationType",
            "Smoking",
            "SmokingYears",
            "SmokingPacksPerDay",
            "UsedToSmokeYear",
            "QuitSmokeYear",
            "UsedToSmokePerDay",
            "Drinking",
            "DrinkingDaysPerWeek",
            "UsedToDrinkYear",
            "UsedToDrinkDayPerWeek",
            "ChronicHTN",
            "ChronicDM",
            "ChronicHLD",
            "InstrumentLearning",
            "InstrumentLearningAge",
            "InstrumentLearningYear",
            "InstrumentType",
            "VideoGamePlaying",
            "VideoGameHourPerWeek",
            "VideoGameController",
            "LanguageDaily", 
            "LanguageNative"
        ], 
        "basic_q_cols": [
            "EHI_Sum", 
            "SF36_PhysicalFunct",
            "SF36_PhysicalLimit",
            "SF36_EmotionalWell",
            "SF36_EmotionalLimit",
            "SF36_Energy",
            "SF36_SocialFunc",
            "SF36_Pain",
            "SF36_GeneralHealth",
            "SF36_Physical",
            "SF36_Mental",
            "PSQI_SleepQuality",
            "PSQI_SleepLatency",
            "PSQI_SleepDuration",
            "PSQI_SleepEfficiency",
            "PSQI_SleepDisturbance",
            "PSQI_SleepMedication",
            "PSQI_DaytimeDysfunc",
            "PSQI_Sum",
            "IPAQ_MET",
            "BFI_Extraversion",
            "BFI_Agreeableness",
            "BFI_Conscientiousness",
            "BFI_EmotionalStability",
            "BFI_Intellect",
            "MSPSS_Sum",
            "CogFailure_Sum",
            "Beck_Anxiety",
            "Beck_Depression"
        ], 
        "structure_cols": [
            "STRUCTURE_NULL_MRI_GM_LEFT_GySulFrontoMargin_VOLUME", 
            "STRUCTURE_NULL_MRI_GM_LEFT_GySulFrontoMargin_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_GySulFrontoMargin_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_GySulOccipitalInf_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_GySulOccipitalInf_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_GySulOccipitalInf_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_GySulParaCentral_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_GySulParaCentral_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_GySulParaCentral_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_GySulSubCentral_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_GySulSubCentral_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_GySulSubCentral_ThickStd", 
            "STRUCTURE_NULL_MRI_GM_LEFT_GySulTransvFrontopol_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_GySulTransvFrontopol_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_GySulTransvFrontopol_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_GySulCingulAnt_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_GySulCingulAnt_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_GySulCingulAnt_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_GySulCingulMidAnt_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_GySulCingulMidAnt_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_GySulCingulMidAnt_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_GySulCingulMidPost_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_GySulCingulMidPost_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_GySulCingulMidPost_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyCingulPostDorsal_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyCingulPostDorsal_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyCingulPostDorsal_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyCingulPostVentral_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyCingulPostVentral_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyCingulPostVentral_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyCuneus_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyCuneus_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyCuneus_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyFrontInfOpercular_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyFrontInfOpercular_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyFrontInfOpercular_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyFrontInfObital_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyFrontInfObital_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyFrontInfObital_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyFrontInfTriangul_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyFrontInfTriangul_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyFrontInfTriangul_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyFrontMiddle_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyFrontMiddle_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyFrontMiddle_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyFrontSup_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyFrontSup_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyFrontSup_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyInsularLongSulCentralInsular_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyInsularLongSulCentralInsular_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyInsularLongSulCentralInsular_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyInsularShort_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyInsularShort_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyInsularShort_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyOccipitalMiddle_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyOccipitalMiddle_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyOccipitalMiddle_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyOccipitalSup_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyOccipitalSup_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyOccipitalSup_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyOccipitalTemporalLateralFusifor_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyOccipitalTemporalLateralFusifor_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyOccipitalTemporalLateralFusifor_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyOccipitalTemporalMedialLingual_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyOccipitalTemporalMedialLingual_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyOccipitalTemporalMedialLingual_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyOccipitalTemporalMedialParahip_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyOccipitalTemporalMedialParahip_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyOccipitalTemporalMedialParahip_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyOrbital_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyOrbital_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyOrbital_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyParietalInfAngular_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyParietalInfAngular_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyParietalInfAngular_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyParietalInfSupramar_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyParietalInfSupramar_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyParietalInfSupramar_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyParietalSuperior_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyParietalSuperior_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyParietalSuperior_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyPostCentral_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyPostCentral_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyPostCentral_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyPreCentral_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyPreCentral_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyPreCentral_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyPreCuneus_VOLUME", 
            "STRUCTURE_NULL_MRI_GM_LEFT_GyPreCuneus_ThickAvg", 
            "STRUCTURE_NULL_MRI_GM_LEFT_GyPreCuneus_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyRectus_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyRectus_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyRectus_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_GySubcallosal_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_GySubcallosal_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_GySubcallosal_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyTemporalSuperiorGyTemporalTransv_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyTemporalSuperiorGyTemporalTransv_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyTemporalSuperiorGyTemporalTransv_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyTemporalSuperiorLateral_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyTemporalSuperiorLateral_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyTemporalSuperiorLateral_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyTemporalSuperiorPlanPolar_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyTemporalSuperiorPlanPolar_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyTemporalSuperiorPlanPolar_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyTemporalSuperiorPlanTempo_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyTemporalSuperiorPlanTempo_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyTemporalSuperiorPlanTempo_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyTemporalInferior_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyTemporalInferior_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyTemporalInferior_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyTemporalMiddle_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyTemporalMiddle_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_GyTemporalMiddle_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_LateralFisAnterorHorizont_VOLUME", 
            "STRUCTURE_NULL_MRI_GM_LEFT_LateralFisAnterorHorizont_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_LateralFisAnterorHorizont_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_LateralFisAnterorVertical_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_LateralFisAnterorVertical_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_LateralFisAnterorVertical_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_LateralFisPost_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_LateralFisPost_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_LateralFisPost_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_PoleOccipital_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_PoleOccipital_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_PoleOccipital_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_PoleTemporal_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_PoleTemporal_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_PoleTemporal_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulCalcarine_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulCalcarine_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulCalcarine_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulCentral_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulCentral_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulCentral_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulCingulMarginalis_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulCingulMarginalis_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulCingulMarginalis_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulCircularInsulaAnteror_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulCircularInsulaAnteror_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulCircularInsulaAnteror_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulCircularInsulaInferior_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulCircularInsulaInferior_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulCircularInsulaInferior_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulCircularInsulaSuperoir_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulCircularInsulaSuperoir_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulCircularInsulaSuperoir_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulCollatTransvAnterior_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulCollatTransvAnterior_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulCollatTransvAnterior_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulCollatTransvPosterior_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulCollatTransvPosterior_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulCollatTransvPosterior_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulFrontInferior_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulFrontInferior_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulFrontInferior_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulFrontMiddle_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulFrontMiddle_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulFrontMiddle_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulFrontSuperior_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulFrontSuperior_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulFrontSuperior_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulIntermPrimJensen_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulIntermPrimJensen_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulIntermPrimJensen_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulIntraParietAndParietalTrans_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulIntraParietAndParietalTrans_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulIntraParietAndParietalTrans_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulOccipitalMiddleAndLunatus_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulOccipitalMiddleAndLunatus_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulOccipitalMiddleAndLunatus_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulOccipitalSuperiorAndTransversal_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulOccipitalSuperiorAndTransversal_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulOccipitalSuperiorAndTransversal_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulOccipitalAnterior_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulOccipitalAnterior_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulOccipitalAnterior_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulOccipitalTemporalLateral_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulOccipitalTemporalLateral_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulOccipitalTemporalLateral_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulOccipitalTemporalMedialAndLingual_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulOccipitalTemporalMedialAndLingual_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulOccipitalTemporalMedialAndLingual_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulOrbitalLateral_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulOrbitalLateral_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulOrbitalLateral_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulOrbitalMedialOlfact_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulOrbitalMedialOlfact_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulOrbitalMedialOlfact_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulOrbitalHshaped_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulOrbitalHshaped_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulOrbitalHshaped_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulParietoOccipital_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulParietoOccipital_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulParietoOccipital_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulPericallosal_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulPericallosal_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulPericallosal_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulPostCentral_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulPostCentral_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulPostCentral_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulPreCentralInferiorPart_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulPreCentralInferiorPart_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulPreCentralInferiorPart_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulPreCentralSuperiorPart_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulPreCentralSuperiorPart_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulPreCentralSuperiorPart_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulSubOrbital_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulSubOrbital_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulSubOrbital_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulSubParietal_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulSubParietal_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulSubParietal_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulTemporalInferior_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulTemporalInferior_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulTemporalInferior_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulTemporalSuperior_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulTemporalSuperior_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulTemporalSuperior_ThickStd",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulTemporalTransverse_VOLUME",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulTemporalTransverse_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_LEFT_SulTemporalTransverse_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GySulFrontoMargin_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GySulFrontoMargin_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GySulFrontoMargin_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GySulOccipitalInf_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GySulOccipitalInf_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GySulOccipitalInf_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GySulParaCentral_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GySulParaCentral_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GySulParaCentral_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GySulSubCentral_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GySulSubCentral_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GySulSubCentral_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GySulTransvFrontopol_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GySulTransvFrontopol_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GySulTransvFrontopol_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GySulCingulAnt_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GySulCingulAnt_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GySulCingulAnt_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GySulCingulMidAnt_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GySulCingulMidAnt_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GySulCingulMidAnt_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GySulCingulMidPost_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GySulCingulMidPost_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GySulCingulMidPost_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyCingulPostDorsal_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyCingulPostDorsal_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyCingulPostDorsal_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyCingulPostVentral_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyCingulPostVentral_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyCingulPostVentral_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyCuneus_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyCuneus_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyCuneus_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyFrontInfOpercular_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyFrontInfOpercular_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyFrontInfOpercular_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyFrontInfObital_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyFrontInfObital_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyFrontInfObital_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyFrontInfTriangul_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyFrontInfTriangul_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyFrontInfTriangul_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyFrontMiddle_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyFrontMiddle_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyFrontMiddle_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyFrontSup_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyFrontSup_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyFrontSup_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyInsularLongSulCentralInsular_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyInsularLongSulCentralInsular_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyInsularLongSulCentralInsular_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyInsularShort_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyInsularShort_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyInsularShort_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyOccipitalMiddle_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyOccipitalMiddle_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyOccipitalMiddle_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyOccipitalSup_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyOccipitalSup_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyOccipitalSup_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyOccipitalTemporalLateralFusifor_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyOccipitalTemporalLateralFusifor_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyOccipitalTemporalLateralFusifor_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyOccipitalTemporalMedialLingual_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyOccipitalTemporalMedialLingual_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyOccipitalTemporalMedialLingual_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyOccipitalTemporalMedialParahip_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyOccipitalTemporalMedialParahip_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyOccipitalTemporalMedialParahip_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyORBITAL_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyORBITAL_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyORBITAL_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyParietalInfAngular_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyParietalInfAngular_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyParietalInfAngular_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyParietalInfSupramar_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyParietalInfSupramar_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyParietalInfSupramar_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyParietalSuperior_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyParietalSuperior_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyParietalSuperior_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyPostCentral_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyPostCentral_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyPostCentral_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyPreCentral_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyPreCentral_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyPreCentral_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyPreCuneus_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyPreCuneus_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyPreCuneus_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyRectus_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyRectus_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyRectus_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GySubcallosal_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GySubcallosal_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GySubcallosal_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyTemporalSuperiorGyTemporalTransv_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyTemporalSuperiorGyTemporalTransv_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyTemporalSuperiorGyTemporalTransv_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyTemporalSuperiorLateral_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyTemporalSuperiorLateral_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyTemporalSuperiorLateral_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyTemporalSuperiorPlanPolar_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyTemporalSuperiorPlanPolar_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyTemporalSuperiorPlanPolar_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyTemporalSuperiorPlanTempo_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyTemporalSuperiorPlanTempo_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyTemporalSuperiorPlanTempo_ThickStd", 
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyTemporalInferior_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyTemporalInferior_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyTemporalInferior_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyTemporalMiddle_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyTemporalMiddle_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_GyTemporalMiddle_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_LateralFisAnterorHorizont_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_LateralFisAnterorHorizont_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_LateralFisAnterorHorizont_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_LateralFisAnterorVertical_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_LateralFisAnterorVertical_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_LateralFisAnterorVertical_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_LateralFisPost_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_LateralFisPost_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_LateralFisPost_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_PoleOccipital_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_PoleOccipital_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_PoleOccipital_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_PoleTemporal_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_PoleTemporal_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_PoleTemporal_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulCalcarine_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulCalcarine_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulCalcarine_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulCentral_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulCentral_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulCentral_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulCingulMarginalis_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulCingulMarginalis_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulCingulMarginalis_ThickStd", 
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulCircularInsulaAnteror_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulCircularInsulaAnteror_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulCircularInsulaAnteror_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulCircularInsulaInferior_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulCircularInsulaInferior_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulCircularInsulaInferior_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulCircularInsulaSuperoir_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulCircularInsulaSuperoir_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulCircularInsulaSuperoir_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulCollatTransvAnterior_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulCollatTransvAnterior_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulCollatTransvAnterior_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulCollatTransvPosterior_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulCollatTransvPosterior_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulCollatTransvPosterior_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulFrontInferior_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulFrontInferior_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulFrontInferior_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulFrontMiddle_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulFrontMiddle_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulFrontMiddle_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulFrontSuperior_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulFrontSuperior_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulFrontSuperior_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulIntermPrimJensen_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulIntermPrimJensen_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulIntermPrimJensen_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulIntraParietAndParietalTrans_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulIntraParietAndParietalTrans_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulIntraParietAndParietalTrans_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulOccipitalMiddleAndLunatus_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulOccipitalMiddleAndLunatus_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulOccipitalMiddleAndLunatus_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulOccipitalSuperiorAndTransversal_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulOccipitalSuperiorAndTransversal_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulOccipitalSuperiorAndTransversal_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulOccipitalAnterior_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulOccipitalAnterior_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulOccipitalAnterior_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulOccipitalTemporalLateral_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulOccipitalTemporalLateral_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulOccipitalTemporalLateral_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulOccipitalTemporalMedialAndLingual_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulOccipitalTemporalMedialAndLingual_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulOccipitalTemporalMedialAndLingual_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulOrbitalLateral_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulOrbitalLateral_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulOrbitalLateral_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulOrbitalMedialOlfact_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulOrbitalMedialOlfact_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulOrbitalMedialOlfact_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulOrbitalHshaped_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulOrbitalHshaped_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulOrbitalHshaped_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulParietoOccipital_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulParietoOccipital_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulParietoOccipital_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulPericallosal_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulPericallosal_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulPericallosal_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulPostCentral_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulPostCentral_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulPostCentral_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulPreCentralInferiorPart_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulPreCentralInferiorPart_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulPreCentralInferiorPart_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulPreCentralSuperiorPart_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulPreCentralSuperiorPart_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulPreCentralSuperiorPart_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulSubOrbital_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulSubOrbital_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulSubOrbital_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulSubParietal_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulSubParietal_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulSubParietal_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulTemporalInferior_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulTemporalInferior_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulTemporalInferior_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulTemporalSuperior_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulTemporalSuperior_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulTemporalSuperior_ThickStd",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulTemporalTransverse_VOLUME",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulTemporalTransverse_ThickAvg",
            "STRUCTURE_NULL_MRI_GM_RIGHT_SulTemporalTransverse_ThickStd",
            "STRUCTURE_NULL_MRI_WM_LEFT_BanksSuperiorTemporalSul_VOLUME",
            "STRUCTURE_NULL_MRI_WM_LEFT_CaudalAnteriorCingulate_VOLUME",
            "STRUCTURE_NULL_MRI_WM_LEFT_CaudalMiddleFrontal_VOLUME",
            "STRUCTURE_NULL_MRI_WM_LEFT_CUNEUS_VOLUME",
            "STRUCTURE_NULL_MRI_WM_LEFT_ENTORHINAL_VOLUME",
            "STRUCTURE_NULL_MRI_WM_LEFT_FUSIFORM_VOLUME",
            "STRUCTURE_NULL_MRI_WM_LEFT_InferiorParietal_VOLUME",
            "STRUCTURE_NULL_MRI_WM_LEFT_InferiorTemporal_VOLUME",
            "STRUCTURE_NULL_MRI_WM_LEFT_IsthmusCingulate_VOLUME",
            "STRUCTURE_NULL_MRI_WM_LEFT_LateralOccipital_VOLUME",
            "STRUCTURE_NULL_MRI_WM_LEFT_LateralOrbitoFrontal_VOLUME",
            "STRUCTURE_NULL_MRI_WM_LEFT_LINGUAL_VOLUME",
            "STRUCTURE_NULL_MRI_WM_LEFT_MedialOrbitoFrontal_VOLUME",
            "STRUCTURE_NULL_MRI_WM_LEFT_MiddleTemporal_VOLUME",
            "STRUCTURE_NULL_MRI_WM_LEFT_ParaHippocampal_VOLUME",
            "STRUCTURE_NULL_MRI_WM_LEFT_ParaCentral_VOLUME",
            "STRUCTURE_NULL_MRI_WM_LEFT_ParsOpercularis_VOLUME",
            "STRUCTURE_NULL_MRI_WM_LEFT_ParsOrbitalis_VOLUME",
            "STRUCTURE_NULL_MRI_WM_LEFT_ParsTriangularis_VOLUME",
            "STRUCTURE_NULL_MRI_WM_LEFT_PERICALCARINE_VOLUME",
            "STRUCTURE_NULL_MRI_WM_LEFT_PostCentral_VOLUME",
            "STRUCTURE_NULL_MRI_WM_LEFT_PosteriorCingulate_VOLUME",
            "STRUCTURE_NULL_MRI_WM_LEFT_PreCentral_VOLUME",
            "STRUCTURE_NULL_MRI_WM_LEFT_PreCuneus_VOLUME",
            "STRUCTURE_NULL_MRI_WM_LEFT_RostralAnteriorCingulate_VOLUME",
            "STRUCTURE_NULL_MRI_WM_LEFT_RostralMiddleFrontal_VOLUME",
            "STRUCTURE_NULL_MRI_WM_LEFT_SuperiorFrontal_VOLUME",
            "STRUCTURE_NULL_MRI_WM_LEFT_SuperiorParietal_VOLUME",
            "STRUCTURE_NULL_MRI_WM_LEFT_SuperiorTemporal_VOLUME",
            "STRUCTURE_NULL_MRI_WM_LEFT_SUPRAMARGINAL_VOLUME",
            "STRUCTURE_NULL_MRI_WM_LEFT_FrontalPole_VOLUME",
            "STRUCTURE_NULL_MRI_WM_LEFT_TemporalPole_VOLUME",
            "STRUCTURE_NULL_MRI_WM_LEFT_TransverseTemporal_VOLUME",
            "STRUCTURE_NULL_MRI_WM_LEFT_INSULA_VOLUME",
            "STRUCTURE_NULL_MRI_WM_RIGHT_BanksSuperiorTemporalSul_VOLUME",
            "STRUCTURE_NULL_MRI_WM_RIGHT_CaudalAnteriorCingulate_VOLUME",
            "STRUCTURE_NULL_MRI_WM_RIGHT_CaudalMiddleFrontal_VOLUME",
            "STRUCTURE_NULL_MRI_WM_RIGHT_CUNEUS_VOLUME",
            "STRUCTURE_NULL_MRI_WM_RIGHT_ENTORHINAL_VOLUME",
            "STRUCTURE_NULL_MRI_WM_RIGHT_FUSIFORM_VOLUME",
            "STRUCTURE_NULL_MRI_WM_RIGHT_InferiorParietal_VOLUME",
            "STRUCTURE_NULL_MRI_WM_RIGHT_InferiorTemporal_VOLUME",
            "STRUCTURE_NULL_MRI_WM_RIGHT_IsthmusCingulate_VOLUME",
            "STRUCTURE_NULL_MRI_WM_RIGHT_LateralOccipital_VOLUME",
            "STRUCTURE_NULL_MRI_WM_RIGHT_LateralOrbitoFrontal_VOLUME",
            "STRUCTURE_NULL_MRI_WM_RIGHT_LINGUAL_VOLUME",
            "STRUCTURE_NULL_MRI_WM_RIGHT_MedialOrbitoFrontal_VOLUME",
            "STRUCTURE_NULL_MRI_WM_RIGHT_MiddleTemporal_VOLUME",
            "STRUCTURE_NULL_MRI_WM_RIGHT_ParaHippocampal_VOLUME",
            "STRUCTURE_NULL_MRI_WM_RIGHT_ParaCentral_VOLUME",
            "STRUCTURE_NULL_MRI_WM_RIGHT_ParsOpercularis_VOLUME",
            "STRUCTURE_NULL_MRI_WM_RIGHT_ParsOrbitalis_VOLUME",
            "STRUCTURE_NULL_MRI_WM_RIGHT_ParsTriangularis_VOLUME",
            "STRUCTURE_NULL_MRI_WM_RIGHT_PERICALCARINE_VOLUME",
            "STRUCTURE_NULL_MRI_WM_RIGHT_PostCentral_VOLUME",
            "STRUCTURE_NULL_MRI_WM_RIGHT_PosteriorCingulate_VOLUME",
            "STRUCTURE_NULL_MRI_WM_RIGHT_PreCentral_VOLUME",
            "STRUCTURE_NULL_MRI_WM_RIGHT_PreCuneus_VOLUME",
            "STRUCTURE_NULL_MRI_WM_RIGHT_RostralAnteriorCingulate_VOLUME",
            "STRUCTURE_NULL_MRI_WM_RIGHT_RostralMiddleFrontal_VOLUME",
            "STRUCTURE_NULL_MRI_WM_RIGHT_SuperiorFrontal_VOLUME",
            "STRUCTURE_NULL_MRI_WM_RIGHT_SuperiorParietal_VOLUME",
            "STRUCTURE_NULL_MRI_WM_RIGHT_SuperiorTemporal_VOLUME",
            "STRUCTURE_NULL_MRI_WM_RIGHT_SUPRAMARGINAL_VOLUME",
            "STRUCTURE_NULL_MRI_WM_RIGHT_FrontalPole_VOLUME",
            "STRUCTURE_NULL_MRI_WM_RIGHT_TemporalPole_VOLUME",
            "STRUCTURE_NULL_MRI_WM_RIGHT_TransverseTemporal_VOLUME",
            "STRUCTURE_NULL_MRI_WM_RIGHT_INSULA_VOLUME"
        ], 
        "motor_cols": [
            "ST_RAW_BPrestHigh",
            "ST_RAW_BPrestLow",
            "ST_RAW_HRrest",
            "ST_RAW_HEIGHT",
            "ST_RAW_WEIGHT",
            "ST_RAW_BMI",
            "ST_RAW_WHR",
            "ST_RAW_AerobicRating",
            "ST_RAW_FunctionalReaching",
            "ST_RAW_WorseLegEyeOpen",
            "ST_RAW_BestLegEyeOpen",
            "ST_RAW_WorseLegEyeClose",
            "ST_RAW_BestLegEyeClose",
            "ST_RAW_PegboardDominant",
            "ST_RAW_PegboardNondominant",
            "ST_RAW_PegboardBoth",
            "ST_RAW_PegboardSum",
            "ST_RAW_PegboardAssemble",
            "ST_RAW_SymbolSearch",
            "ST_RAW_SymbolCoding",
            "ST_RAW_GripMvcL",
            "ST_RAW_GripMvcR",
            "ST_NORM_BMI",
            "ST_NORM_WHR",
            "ST_NORM_AerobicRating",
            "ST_NORM_FunctionalReaching",
            "ST_NORM_BestLegEyeClose",
            "ST_NORM_PegboardDominant",
            "ST_NORM_PegboardNondominant",
            "ST_NORM_PegboardSum",
            "ST_NORM_PegboardBoth",
            "ST_NORM_PegboardAssemble",
            "ST_NORM_WaisSpeed",
            "ST_SCALED_FineMotor",
            "ST_SCALED_Balance",
            "ST_SCALED_ProcessingSpeed",
            "GFORCE_BEH_LeftLowForce_RT",
            "GFORCE_BEH_LeftLowForce_RTsd",
            "GFORCE_BEH_LeftLowForce_ACC",
            "GFORCE_BEH_LeftLowForce_ACCsd",
            "GFORCE_BEH_LeftHighForce_RT",
            "GFORCE_BEH_LeftHighForce_RTsd",
            "GFORCE_BEH_LeftHighForce_ACC",
            "GFORCE_BEH_LeftHighForce_ACCsd",
            "GFORCE_BEH_RightLowForce_RT",
            "GFORCE_BEH_RightLowForce_RTsd",
            "GFORCE_BEH_RightLowForce_ACC",
            "GFORCE_BEH_RightLowForce_ACCsd",
            "GFORCE_BEH_RightHighForce_RT",
            "GFORCE_BEH_RightHighForce_RTsd",
            "GFORCE_BEH_RightHighForce_ACC",
            "GFORCE_BEH_RightHighForce_ACCsd",
            "GFORCE_MRI_LeftHighForce_LH_CerebVVI_BETA",
            "GFORCE_MRI_LeftHighForce_LH_PMC_BETA",
            "GFORCE_MRI_LeftHighForce_LH_SMA_BETA",
            "GFORCE_MRI_LeftHighForce_LH_insula_BETA",
            "GFORCE_MRI_LeftHighForce_RH_CerebVVI_BETA",
            "GFORCE_MRI_LeftHighForce_RH_PMC_BETA",
            "GFORCE_MRI_LeftHighForce_RH_SMA_BETA",
            "GFORCE_MRI_LeftHighForce_RH_insula_BETA",
            "GFORCE_MRI_LeftLowForce_LH_CerebVVI_BETA",
            "GFORCE_MRI_LeftLowForce_LH_PMC_BETA",
            "GFORCE_MRI_LeftLowForce_LH_SMA_BETA",
            "GFORCE_MRI_LeftLowForce_LH_insula_BETA",
            "GFORCE_MRI_LeftLowForce_RH_CerebVVI_BETA",
            "GFORCE_MRI_LeftLowForce_RH_PMC_BETA",
            "GFORCE_MRI_LeftLowForce_RH_SMA_BETA",
            "GFORCE_MRI_LeftLowForce_RH_insula_BETA",
            "GFORCE_MRI_RightHighForce_LH_CerebVVI_BETA",
            "GFORCE_MRI_RightHighForce_LH_PMC_BETA",
            "GFORCE_MRI_RightHighForce_LH_SMA_BETA",
            "GFORCE_MRI_RightHighForce_LH_insula_BETA",
            "GFORCE_MRI_RightHighForce_RH_CerebVVI_BETA",
            "GFORCE_MRI_RightHighForce_RH_PMC_BETA",
            "GFORCE_MRI_RightHighForce_RH_SMA_BETA",
            "GFORCE_MRI_RightHighForce_RH_insula_BETA",
            "GFORCE_MRI_RightLowForce_LH_CerebVVI_BETA",
            "GFORCE_MRI_RightLowForce_LH_PMC_BETA",
            "GFORCE_MRI_RightLowForce_LH_SMA_BETA",
            "GFORCE_MRI_RightLowForce_LH_insula_BETA",
            "GFORCE_MRI_RightLowForce_RH_CerebVVI_BETA",
            "GFORCE_MRI_RightLowForce_RH_PMC_BETA",
            "GFORCE_MRI_RightLowForce_RH_SMA_BETA",
            "GFORCE_MRI_RightLowForce_RH_insula_BETA",
            "GFORCE_MRI_LeftHighForce_LH_CerebVVI_VoxelCount",
            "GFORCE_MRI_LeftHighForce_LH_PMC_VoxelCount",
            "GFORCE_MRI_LeftHighForce_LH_SMA_VoxelCount",
            "GFORCE_MRI_LeftHighForce_LH_insula_VoxelCount",
            "GFORCE_MRI_LeftHighForce_RH_CerebVVI_VoxelCount",
            "GFORCE_MRI_LeftHighForce_RH_PMC_VoxelCount",
            "GFORCE_MRI_LeftHighForce_RH_SMA_VoxelCount",
            "GFORCE_MRI_LeftHighForce_RH_insula_VoxelCount",
            "GFORCE_MRI_LeftLowForce_LH_CerebVVI_VoxelCount",
            "GFORCE_MRI_LeftLowForce_LH_PMC_VoxelCount",
            "GFORCE_MRI_LeftLowForce_LH_SMA_VoxelCount",
            "GFORCE_MRI_LeftLowForce_LH_insula_VoxelCount",
            "GFORCE_MRI_LeftLowForce_RH_CerebVVI_VoxelCount",
            "GFORCE_MRI_LeftLowForce_RH_PMC_VoxelCount",
            "GFORCE_MRI_LeftLowForce_RH_SMA_VoxelCount",
            "GFORCE_MRI_LeftLowForce_RH_insula_VoxelCount",
            "GFORCE_MRI_RightHighForce_LH_CerebVVI_VoxelCount",
            "GFORCE_MRI_RightHighForce_LH_PMC_VoxelCount",
            "GFORCE_MRI_RightHighForce_LH_SMA_VoxelCount",
            "GFORCE_MRI_RightHighForce_LH_insula_VoxelCount",
            "GFORCE_MRI_RightHighForce_RH_CerebVVI_VoxelCount",
            "GFORCE_MRI_RightHighForce_RH_PMC_VoxelCount",
            "GFORCE_MRI_RightHighForce_RH_SMA_VoxelCount",
            "GFORCE_MRI_RightHighForce_RH_insula_VoxelCount",
            "GFORCE_MRI_RightLowForce_LH_CerebVVI_VoxelCount",
            "GFORCE_MRI_RightLowForce_LH_PMC_VoxelCount",
            "GFORCE_MRI_RightLowForce_LH_SMA_VoxelCount",
            "GFORCE_MRI_RightLowForce_LH_insula_VoxelCount",
            "GFORCE_MRI_RightLowForce_RH_CerebVVI_VoxelCount",
            "GFORCE_MRI_RightLowForce_RH_PMC_VoxelCount",
            "GFORCE_MRI_RightLowForce_RH_SMA_VoxelCount",
            "GFORCE_MRI_RightLowForce_RH_insula_VoxelCount",
            "GOFITTS_EEG_CueSlope_P3_N200Time_NULL",
            "GOFITTS_EEG_CueSlope_P3_N200Amp_NULL",
            "GOFITTS_EEG_CueSlope_P3_P300Time_NULL",
            "GOFITTS_EEG_CueSlope_P3_P300Amp_NULL",
            "GOFITTS_EEG_CueSlope_P3_DiffTime_NULL",
            "GOFITTS_EEG_CueSlope_P3_DiffAmp_NULL",
            "GOFITTS_EEG_CueSlopeW3_P3_N200Time_NULL",
            "GOFITTS_EEG_CueSlopeW3_P3_N200Amp_NULL",
            "GOFITTS_EEG_CueSlopeW3_P3_P300Time_NULL",
            "GOFITTS_EEG_CueSlopeW3_P3_P300Amp_NULL",
            "GOFITTS_EEG_CueSlopeW3_P3_DiffTime_NULL",
            "GOFITTS_EEG_CueSlopeW3_P3_DiffAmp_NULL",
            "GOFITTS_EEG_CueSlopeW2_P3_N200Time_NULL",
            "GOFITTS_EEG_CueSlopeW2_P3_N200Amp_NULL",
            "GOFITTS_EEG_CueSlopeW2_P3_P300Time_NULL",
            "GOFITTS_EEG_CueSlopeW2_P3_P300Amp_NULL",
            "GOFITTS_EEG_CueSlopeW2_P3_DiffTime_NULL",
            "GOFITTS_EEG_CueSlopeW2_P3_DiffAmp_NULL",
            "GOFITTS_EEG_CueSlope_P4_N200Time_NULL",
            "GOFITTS_EEG_CueSlope_P4_N200Amp_NULL",
            "GOFITTS_EEG_CueSlope_P4_P300Time_NULL",
            "GOFITTS_EEG_CueSlope_P4_P300Amp_NULL",
            "GOFITTS_EEG_CueSlope_P4_DiffTime_NULL",
            "GOFITTS_EEG_CueSlope_P4_DiffAmp_NULL",
            "GOFITTS_EEG_CueSlopeW3_P4_N200Time_NULL",
            "GOFITTS_EEG_CueSlopeW3_P4_N200Amp_NULL",
            "GOFITTS_EEG_CueSlopeW3_P4_P300Time_NULL",
            "GOFITTS_EEG_CueSlopeW3_P4_P300Amp_NULL",
            "GOFITTS_EEG_CueSlopeW3_P4_DiffTime_NULL",
            "GOFITTS_EEG_CueSlopeW3_P4_DiffAmp_NULL",
            "GOFITTS_EEG_CueSlopeW2_P4_N200Time_NULL",
            "GOFITTS_EEG_CueSlopeW2_P4_N200Amp_NULL",
            "GOFITTS_EEG_CueSlopeW2_P4_P300Time_NULL",
            "GOFITTS_EEG_CueSlopeW2_P4_P300Amp_NULL",
            "GOFITTS_EEG_CueSlopeW2_P4_DiffTime_NULL",
            "GOFITTS_EEG_CueSlopeW2_P4_DiffAmp_NULL",
            "GOFITTS_EEG_GoSlope_P3_N200Time_NULL",
            "GOFITTS_EEG_GoSlope_P3_N200Amp_NULL",
            "GOFITTS_EEG_GoSlope_P3_P300Time_NULL",
            "GOFITTS_EEG_GoSlope_P3_P300Amp_NULL",
            "GOFITTS_EEG_GoSlope_P3_DiffTime_NULL",
            "GOFITTS_EEG_GoSlope_P3_DiffAmp_NULL",
            "GOFITTS_EEG_GoSlopeW3_P3_N200Time_NULL",
            "GOFITTS_EEG_GoSlopeW3_P3_N200Amp_NULL",
            "GOFITTS_EEG_GoSlopeW3_P3_P300Time_NULL",
            "GOFITTS_EEG_GoSlopeW3_P3_P300Amp_NULL",
            "GOFITTS_EEG_GoSlopeW3_P3_DiffTime_NULL",
            "GOFITTS_EEG_GoSlopeW3_P3_DiffAmp_NULL",
            "GOFITTS_EEG_GoSlopeW2_P3_N200Time_NULL",
            "GOFITTS_EEG_GoSlopeW2_P3_N200Amp_NULL",
            "GOFITTS_EEG_GoSlopeW2_P3_P300Time_NULL",
            "GOFITTS_EEG_GoSlopeW2_P3_P300Amp_NULL",
            "GOFITTS_EEG_GoSlopeW2_P3_DiffTime_NULL",
            "GOFITTS_EEG_GoSlopeW2_P3_DiffAmp_NULL",
            "GOFITTS_EEG_GoSlope_P4_N200Time_NULL",
            "GOFITTS_EEG_GoSlope_P4_N200Amp_NULL",
            "GOFITTS_EEG_GoSlope_P4_P300Time_NULL",
            "GOFITTS_EEG_GoSlope_P4_P300Amp_NULL",
            "GOFITTS_EEG_GoSlope_P4_DiffTime_NULL",
            "GOFITTS_EEG_GoSlope_P4_DiffAmp_NULL",
            "GOFITTS_EEG_GoSlopeW3_P4_N200Time_NULL",
            "GOFITTS_EEG_GoSlopeW3_P4_N200Amp_NULL",
            "GOFITTS_EEG_GoSlopeW3_P4_P300Time_NULL",
            "GOFITTS_EEG_GoSlopeW3_P4_P300Amp_NULL",
            "GOFITTS_EEG_GoSlopeW3_P4_DiffTime_NULL",
            "GOFITTS_EEG_GoSlopeW3_P4_DiffAmp_NULL",
            "GOFITTS_EEG_GoSlopeW2_P4_N200Time_NULL",
            "GOFITTS_EEG_GoSlopeW2_P4_N200Amp_NULL",
            "GOFITTS_EEG_GoSlopeW2_P4_P300Time_NULL",
            "GOFITTS_EEG_GoSlopeW2_P4_P300Amp_NULL",
            "GOFITTS_EEG_GoSlopeW2_P4_DiffTime_NULL",
            "GOFITTS_EEG_GoSlopeW2_P4_DiffAmp_NULL",
            "GOFITTS_EEG_CueID1_P3_N200Time_NULL",
            "GOFITTS_EEG_CueID2_P3_N200Time_NULL",
            "GOFITTS_EEG_CueID3_P3_N200Time_NULL",
            "GOFITTS_EEG_CueID4_P3_N200Time_NULL",
            "GOFITTS_EEG_CueID5_P3_N200Time_NULL",
            "GOFITTS_EEG_CueID6_P3_N200Time_NULL",
            "GOFITTS_EEG_CueID1_P3_N200Amp_NULL",
            "GOFITTS_EEG_CueID2_P3_N200Amp_NULL",
            "GOFITTS_EEG_CueID3_P3_N200Amp_NULL",
            "GOFITTS_EEG_CueID4_P3_N200Amp_NULL",
            "GOFITTS_EEG_CueID5_P3_N200Amp_NULL",
            "GOFITTS_EEG_CueID6_P3_N200Amp_NULL",
            "GOFITTS_EEG_CueID1_P3_P300Time_NULL",
            "GOFITTS_EEG_CueID2_P3_P300Time_NULL",
            "GOFITTS_EEG_CueID3_P3_P300Time_NULL",
            "GOFITTS_EEG_CueID4_P3_P300Time_NULL",
            "GOFITTS_EEG_CueID5_P3_P300Time_NULL",
            "GOFITTS_EEG_CueID6_P3_P300Time_NULL",
            "GOFITTS_EEG_CueID1_P3_P300Amp_NULL",
            "GOFITTS_EEG_CueID2_P3_P300Amp_NULL",
            "GOFITTS_EEG_CueID3_P3_P300Amp_NULL",
            "GOFITTS_EEG_CueID4_P3_P300Amp_NULL",
            "GOFITTS_EEG_CueID5_P3_P300Amp_NULL",
            "GOFITTS_EEG_CueID6_P3_P300Amp_NULL",
            "GOFITTS_EEG_CueID1_P3_DiffTime_NULL",
            "GOFITTS_EEG_CueID2_P3_DiffTime_NULL",
            "GOFITTS_EEG_CueID3_P3_DiffTime_NULL",
            "GOFITTS_EEG_CueID4_P3_DiffTime_NULL",
            "GOFITTS_EEG_CueID5_P3_DiffTime_NULL",
            "GOFITTS_EEG_CueID6_P3_DiffTime_NULL",
            "GOFITTS_EEG_CueID1_P3_DiffAmp_NULL",
            "GOFITTS_EEG_CueID2_P3_DiffAmp_NULL",
            "GOFITTS_EEG_CueID3_P3_DiffAmp_NULL",
            "GOFITTS_EEG_CueID4_P3_DiffAmp_NULL",
            "GOFITTS_EEG_CueID5_P3_DiffAmp_NULL",
            "GOFITTS_EEG_CueID6_P3_DiffAmp_NULL",
            "GOFITTS_EEG_CueID1_P4_N200Time_NULL",
            "GOFITTS_EEG_CueID2_P4_N200Time_NULL",
            "GOFITTS_EEG_CueID3_P4_N200Time_NULL",
            "GOFITTS_EEG_CueID4_P4_N200Time_NULL",
            "GOFITTS_EEG_CueID5_P4_N200Time_NULL",
            "GOFITTS_EEG_CueID6_P4_N200Time_NULL",
            "GOFITTS_EEG_CueID1_P4_N200Amp_NULL",
            "GOFITTS_EEG_CueID2_P4_N200Amp_NULL",
            "GOFITTS_EEG_CueID3_P4_N200Amp_NULL",
            "GOFITTS_EEG_CueID4_P4_N200Amp_NULL",
            "GOFITTS_EEG_CueID5_P4_N200Amp_NULL",
            "GOFITTS_EEG_CueID6_P4_N200Amp_NULL",
            "GOFITTS_EEG_CueID1_P4_P300Time_NULL",
            "GOFITTS_EEG_CueID2_P4_P300Time_NULL",
            "GOFITTS_EEG_CueID3_P4_P300Time_NULL",
            "GOFITTS_EEG_CueID4_P4_P300Time_NULL",
            "GOFITTS_EEG_CueID5_P4_P300Time_NULL",
            "GOFITTS_EEG_CueID6_P4_P300Time_NULL",
            "GOFITTS_EEG_CueID1_P4_P300Amp_NULL",
            "GOFITTS_EEG_CueID2_P4_P300Amp_NULL",
            "GOFITTS_EEG_CueID3_P4_P300Amp_NULL",
            "GOFITTS_EEG_CueID4_P4_P300Amp_NULL",
            "GOFITTS_EEG_CueID5_P4_P300Amp_NULL",
            "GOFITTS_EEG_CueID6_P4_P300Amp_NULL",
            "GOFITTS_EEG_CueID1_P4_DiffTime_NULL",
            "GOFITTS_EEG_CueID2_P4_DiffTime_NULL",
            "GOFITTS_EEG_CueID3_P4_DiffTime_NULL",
            "GOFITTS_EEG_CueID4_P4_DiffTime_NULL",
            "GOFITTS_EEG_CueID5_P4_DiffTime_NULL",
            "GOFITTS_EEG_CueID6_P4_DiffTime_NULL",
            "GOFITTS_EEG_CueID1_P4_DiffAmp_NULL",
            "GOFITTS_EEG_CueID2_P4_DiffAmp_NULL",
            "GOFITTS_EEG_CueID3_P4_DiffAmp_NULL",
            "GOFITTS_EEG_CueID4_P4_DiffAmp_NULL",
            "GOFITTS_EEG_CueID5_P4_DiffAmp_NULL",
            "GOFITTS_EEG_CueID6_P4_DiffAmp_NULL",
            "GOFITTS_EEG_GoID1_P3_N200Time_NULL",
            "GOFITTS_EEG_GoID2_P3_N200Time_NULL",
            "GOFITTS_EEG_GoID3_P3_N200Time_NULL",
            "GOFITTS_EEG_GoID4_P3_N200Time_NULL",
            "GOFITTS_EEG_GoID5_P3_N200Time_NULL",
            "GOFITTS_EEG_GoID6_P3_N200Time_NULL",
            "GOFITTS_EEG_GoID1_P3_N200Amp_NULL",
            "GOFITTS_EEG_GoID2_P3_N200Amp_NULL",
            "GOFITTS_EEG_GoID3_P3_N200Amp_NULL",
            "GOFITTS_EEG_GoID4_P3_N200Amp_NULL",
            "GOFITTS_EEG_GoID5_P3_N200Amp_NULL",
            "GOFITTS_EEG_GoID6_P3_N200Amp_NULL",
            "GOFITTS_EEG_GoID1_P3_P300Time_NULL",
            "GOFITTS_EEG_GoID2_P3_P300Time_NULL",
            "GOFITTS_EEG_GoID3_P3_P300Time_NULL",
            "GOFITTS_EEG_GoID4_P3_P300Time_NULL",
            "GOFITTS_EEG_GoID5_P3_P300Time_NULL",
            "GOFITTS_EEG_GoID6_P3_P300Time_NULL",
            "GOFITTS_EEG_GoID1_P3_P300Amp_NULL",
            "GOFITTS_EEG_GoID2_P3_P300Amp_NULL",
            "GOFITTS_EEG_GoID3_P3_P300Amp_NULL",
            "GOFITTS_EEG_GoID4_P3_P300Amp_NULL",
            "GOFITTS_EEG_GoID5_P3_P300Amp_NULL",
            "GOFITTS_EEG_GoID6_P3_P300Amp_NULL",
            "GOFITTS_EEG_GoID1_P3_DiffTime_NULL",
            "GOFITTS_EEG_GoID2_P3_DiffTime_NULL",
            "GOFITTS_EEG_GoID3_P3_DiffTime_NULL",
            "GOFITTS_EEG_GoID4_P3_DiffTime_NULL",
            "GOFITTS_EEG_GoID5_P3_DiffTime_NULL",
            "GOFITTS_EEG_GoID6_P3_DiffTime_NULL",
            "GOFITTS_EEG_GoID1_P3_DiffAmp_NULL",
            "GOFITTS_EEG_GoID2_P3_DiffAmp_NULL",
            "GOFITTS_EEG_GoID3_P3_DiffAmp_NULL",
            "GOFITTS_EEG_GoID4_P3_DiffAmp_NULL",
            "GOFITTS_EEG_GoID5_P3_DiffAmp_NULL",
            "GOFITTS_EEG_GoID6_P3_DiffAmp_NULL",
            "GOFITTS_EEG_GoID1_P4_N200Time_NULL",
            "GOFITTS_EEG_GoID2_P4_N200Time_NULL",
            "GOFITTS_EEG_GoID3_P4_N200Time_NULL",
            "GOFITTS_EEG_GoID4_P4_N200Time_NULL",
            "GOFITTS_EEG_GoID5_P4_N200Time_NULL",
            "GOFITTS_EEG_GoID6_P4_N200Time_NULL",
            "GOFITTS_EEG_GoID1_P4_N200Amp_NULL",
            "GOFITTS_EEG_GoID2_P4_N200Amp_NULL",
            "GOFITTS_EEG_GoID3_P4_N200Amp_NULL",
            "GOFITTS_EEG_GoID4_P4_N200Amp_NULL",
            "GOFITTS_EEG_GoID5_P4_N200Amp_NULL",
            "GOFITTS_EEG_GoID6_P4_N200Amp_NULL",
            "GOFITTS_EEG_GoID1_P4_P300Time_NULL",
            "GOFITTS_EEG_GoID2_P4_P300Time_NULL",
            "GOFITTS_EEG_GoID3_P4_P300Time_NULL",
            "GOFITTS_EEG_GoID4_P4_P300Time_NULL",
            "GOFITTS_EEG_GoID5_P4_P300Time_NULL",
            "GOFITTS_EEG_GoID6_P4_P300Time_NULL",
            "GOFITTS_EEG_GoID1_P4_P300Amp_NULL",
            "GOFITTS_EEG_GoID2_P4_P300Amp_NULL",
            "GOFITTS_EEG_GoID3_P4_P300Amp_NULL",
            "GOFITTS_EEG_GoID4_P4_P300Amp_NULL",
            "GOFITTS_EEG_GoID5_P4_P300Amp_NULL",
            "GOFITTS_EEG_GoID6_P4_P300Amp_NULL",
            "GOFITTS_EEG_GoID1_P4_DiffTime_NULL",
            "GOFITTS_EEG_GoID2_P4_DiffTime_NULL",
            "GOFITTS_EEG_GoID3_P4_DiffTime_NULL",
            "GOFITTS_EEG_GoID4_P4_DiffTime_NULL",
            "GOFITTS_EEG_GoID5_P4_DiffTime_NULL",
            "GOFITTS_EEG_GoID6_P4_DiffTime_NULL",
            "GOFITTS_EEG_GoID1_P4_DiffAmp_NULL",
            "GOFITTS_EEG_GoID2_P4_DiffAmp_NULL",
            "GOFITTS_EEG_GoID3_P4_DiffAmp_NULL",
            "GOFITTS_EEG_GoID4_P4_DiffAmp_NULL",
            "GOFITTS_EEG_GoID5_P4_DiffAmp_NULL",
            "GOFITTS_EEG_GoID6_P4_DiffAmp_NULL",
            "GOFITTS_BEH_ID1_LeaveTime",
            "GOFITTS_BEH_ID2_LeaveTime",
            "GOFITTS_BEH_ID3_LeaveTime",
            "GOFITTS_BEH_ID4_LeaveTime",
            "GOFITTS_BEH_ID5_LeaveTime",
            "GOFITTS_BEH_ID6_LeaveTime",
            "GOFITTS_BEH_ID1_PointTime",
            "GOFITTS_BEH_ID2_PointTime",
            "GOFITTS_BEH_ID3_PointTime",
            "GOFITTS_BEH_ID4_PointTime",
            "GOFITTS_BEH_ID5_PointTime",
            "GOFITTS_BEH_ID6_PointTime",
            "GOFITTS_BEH_ID1_Throughput",
            "GOFITTS_BEH_ID2_Throughput",
            "GOFITTS_BEH_ID3_Throughput",
            "GOFITTS_BEH_ID4_Throughput",
            "GOFITTS_BEH_ID5_Throughput",
            "GOFITTS_BEH_ID6_Throughput",
            "GOFITTS_BEH_SLOPE_LeaveTime",
            "GOFITTS_BEH_SLOPE_PointTime",
            "GOFITTS_BEH_SLOPE_LeaveTimeW3",
            "GOFITTS_BEH_SLOPE_LeaveTimeW2",
            "GOFITTS_BEH_SLOPE_PointTimeW3",
            "GOFITTS_BEH_SLOPE_PointTimeW2",
            "BILPRESS_BEH_HighForceSyncing_ReachTime",
            "BILPRESS_BEH_HighForceSyncing_VARIABILITY",
            "BILPRESS_BEH_HighForceStable_VARIABILITY",
            "BILPRESS_BEH_LowForceSyncing_ReachTime",
            "BILPRESS_BEH_LowForceSyncing_VARIABILITY",
            "BILPRESS_BEH_LowForceStable_VARIABILITY",
            "BILPRESS_EEG_HighForceStable_C3_NULL_ALPHA",
            "BILPRESS_EEG_HighForceStable_C3_NULL_BETA",
            "BILPRESS_EEG_HighForceStable_C4_NULL_ALPHA",
            "BILPRESS_EEG_HighForceStable_C4_NULL_BETA",
            "BILPRESS_EEG_HighForceSyncing_C3_NULL_ALPHA",
            "BILPRESS_EEG_HighForceSyncing_C3_NULL_BETA",
            "BILPRESS_EEG_HighForceSyncing_C4_NULL_ALPHA",
            "BILPRESS_EEG_HighForceSyncing_C4_NULL_BETA",
            "BILPRESS_EEG_LowForceStable_C3_NULL_ALPHA",
            "BILPRESS_EEG_LowForceStable_C3_NULL_BETA",
            "BILPRESS_EEG_LowForceStable_C4_NULL_ALPHA",
            "BILPRESS_EEG_LowForceStable_C4_NULL_BETA",
            "BILPRESS_EEG_LowForceSyncing_C3_NULL_ALPHA",
            "BILPRESS_EEG_LowForceSyncing_C3_NULL_BETA",
            "BILPRESS_EEG_LowForceSyncing_C4_NULL_ALPHA",
            "BILPRESS_EEG_LowForceSyncing_C4_NULL_BETA",
            "BILPRESS_EEG_LowForceSyncing_C3C4Diff_NULL_ALPHA",
            "BILPRESS_EEG_HighForceSyncing_C3C4Diff_NULL_ALPHA",
            "BILPRESS_EEG_LowForceSyncing_C3C4Diff_NULL_BETA",
            "BILPRESS_EEG_HighForceSyncing_C3C4Diff_NULL_BETA",
            "BILPRESS_EEG_LowForceStable_C3C4Diff_NULL_ALPHA",
            "BILPRESS_EEG_HighForceStable_C3C4Diff_NULL_ALPHA",
            "BILPRESS_EEG_LowForceStable_C3C4Diff_NULL_BETA",
            "BILPRESS_EEG_HighForceStable_C3C4Diff_NULL_BETA",
            "BILPRESS_EEG_HighForceStable_PSI_NULL_ALPHA",
            "BILPRESS_EEG_HighForceStable_PSI_NULL_BETA",
            "BILPRESS_EEG_HighForceSyncing_PSI_NULL_ALPHA",
            "BILPRESS_EEG_HighForceSyncing_PSI_NULL_BETA",
            "BILPRESS_EEG_LowForceStable_PSI_NULL_ALPHA",
            "BILPRESS_EEG_LowForceStable_PSI_NULL_BETA",
            "BILPRESS_EEG_LowForceSyncing_PSI_NULL_ALPHA",
            "BILPRESS_EEG_LowForceSyncing_PSI_NULL_BETA",
        ], 
        "language_cols": [
            "ST_RAW_SIMILARITY",
            "ST_RAW_VOCABULARY",
            "ST_RAW_INFORMATION",
            "ST_RAW_SUM",
            "ST_SCALED_SIMILARITY",
            "ST_SCALED_VOCABULARY",
            "ST_SCALED_INFORMATION",
            "ST_SCALED_SUM",
            "ST_NORM_SIMILARITY",
            "ST_NORM_VOCABULARY",
            "ST_NORM_INFORMATION",
            "ST_NORM_VCI",
            "ST_NORM_PR",
            "READING_BEH_NULL_MeanSR",
            "STORY_BEH_NULL_MeanSR",
            "WORDNAME_BEH_HFHC_ACCURACY",
            "WORDNAME_BEH_HFLC_ACCURACY",
            "WORDNAME_BEH_LFHC_ACCURACY",
            "WORDNAME_BEH_LFLC_ACCURACY",
            "SPEECHCOMP_BEH_ACTION_ACCURACY",
            "SPEECHCOMP_BEH_OBJECT_ACCURACY",
            "SPEECHCOMP_BEH_PASSIVE_ACCURACY",
            "SPEECHCOMP_BEH_CONTROL_ACCURACY",
            "WORDNAME_BEH_HFHC_RT",
            "WORDNAME_BEH_HFLC_RT",
            "WORDNAME_BEH_LFHC_RT",
            "WORDNAME_BEH_LFLC_RT",
            "WORDNAME_MRI_HFHC_BROCA_BETA",
            "WORDNAME_MRI_HFLC_BROCA_BETA",
            "WORDNAME_MRI_LFHC_BROCA_BETA",
            "WORDNAME_MRI_LFLC_BROCA_BETA",
            "WORDNAME_MRI_DiffLFHF_BROCA_BETA",
            "WORDNAME_MRI_DiffLCHC_BROCA_BETA",
            "SPEECHCOMP_MRI_ACTION_BROCA_BETA",
            "SPEECHCOMP_MRI_OBJECT_BROCA_BETA",
            "SPEECHCOMP_MRI_PASSIVE_BROCA_BETA",
            "SPEECHCOMP_MRI_CONTROL_BROCA_BETA",
            "SPEECHCOMP_MRI_DiffAC_BROCA_BETA",
            "SPEECHCOMP_MRI_DiffOC_BROCA_BETA",
            "SPEECHCOMP_MRI_DiffPC_BROCA_BETA",
            "WORDNAME_MRI_HFHC_WERNICKE_BETA",
            "WORDNAME_MRI_HFLC_WERNICKE_BETA",
            "WORDNAME_MRI_LFHC_WERNICKE_BETA",
            "WORDNAME_MRI_LFLC_WERNICKE_BETA",
            "WORDNAME_MRI_DiffLFHF_WERNICKE_BETA",
            "WORDNAME_MRI_DiffLCHC_WERNICKE_BETA",
            "SPEECHCOMP_MRI_ACTION_WERNICKE_BETA",
            "SPEECHCOMP_MRI_OBJECT_WERNICKE_BETA",
            "SPEECHCOMP_MRI_PASSIVE_WERNICKE_BETA",
            "SPEECHCOMP_MRI_CONTROL_WERNICKE_BETA",
            "SPEECHCOMP_MRI_DiffAC_WERNICKE_BETA",
            "SPEECHCOMP_MRI_DiffOC_WERNICKE_BETA",
            "SPEECHCOMP_MRI_DiffPC_WERNICKE_BETA",
            "WORDNAME_MRI_HFHC_LeftAntTemporalPole_BETA",
            "WORDNAME_MRI_HFLC_LeftAntTemporalPole_BETA",
            "WORDNAME_MRI_LFHC_LeftAntTemporalPole_BETA",
            "WORDNAME_MRI_LFLC_LeftAntTemporalPole_BETA",
            "WORDNAME_MRI_DiffLFHF_LeftAntTemporalPole_BETA",
            "WORDNAME_MRI_DiffLCHC_LeftAntTemporalPole_BETA",
            "SPEECHCOMP_MRI_ACTION_LeftAntTemporalPole_BETA",
            "SPEECHCOMP_MRI_OBJECT_LeftAntTemporalPole_BETA",
            "SPEECHCOMP_MRI_PASSIVE_LeftAntTemporalPole_BETA",
            "SPEECHCOMP_MRI_CONTROL_LeftAntTemporalPole_BETA",
            "SPEECHCOMP_MRI_DiffAC_LeftAntTemporalPole_BETA",
            "SPEECHCOMP_MRI_DiffOC_LeftAntTemporalPole_BETA",
            "SPEECHCOMP_MRI_DiffPC_LeftAntTemporalPole_BETA",
        ]
    }[name]


def print_missing(all_subjs: list[str], data_subjs: list[str]):
    missing = [ s for s in all_subjs if s not in data_subjs ]

    if missing:
        print(f"\n[Warning] data of {len(missing)} participant(s) are missing:")
        for m in missing:
            print(f"\t- {m}")


def load_img_data(
    img_path: [str | Path], 
    print_infos: bool = False, 
    get_affine: bool = False
) -> np.ndarray:
    '''
    Load the image and return its array data
    If `print_infos` is True, print specified information about the image
    If `get_affine` is True, returns image affine as well
    '''
    def _print_infos(img: nib.nifti1.Nifti1Image):
        hdr = img.header
        print(f"Image shape: {img.shape}")
        print(f"Image orientations: {nib.aff2axcodes(img.affine)}")
        print({
            0: "sform not defined", 
            1: "RAS+ in scanner coordinates", 
            2: "RAS+ aligned to some other scan", 
            3: "RAS+ in Talairach atlas space", 
            4: "RAS+ in MNI atlas space"
        }[hdr["sform_code"]])
        print(f'Description: {hdr["descrip"]}')
        print()

    img_path = str(img_path)
    assert os.path.isfile(img_path), f"Image file not exists: {img_path}"

    print(f"Loading image: {img_path}")
    img = nib.load(img_path)

    if print_infos:
        _print_infos(img)

    print("Loading the array data ...")
    img_dat = np.asarray(img.dataobj, dtype=np.float32)  # avoid caching
        # see: https://nipy.org/nibabel/images_and_memory.html#use-the-array-proxy-instead-of-get-fdata

    if get_affine:
        return img_dat, img.affine
    else:
        return img_dat
    

def get_tbss_processed(
    img_path: [str | Path], 
    mask_path: [str | Path], 
    cache: [str | Path], 
    N: int, 
    stride: int, 
    **kwargs
) -> np.ndarray:
    '''
    Load the 4-D TBSS stack, downsample it by stride, 
    extract in-mask voxels and flatten it per participant, 
    and z-score with global mean and SD

    Returns: (N, n_vox) float32
    '''
    if cache.exists():
        print(f"Loaded from cache: {cache}")
        return np.load(cache, mmap_mode="r")

    img_dat = load_img_data(img_path)
    N_v = img_dat.shape[-1]
    assert N_v == N, f"Mismatch between volume ({N_v}) and globbed ({N}) subject count."
    
    img_dat = img_dat[::stride, ::stride, ::stride, :]
    img_dat = np.moveaxis(img_dat, -1, 0)  # (N, X, Y, Z)

    mask_dat = load_img_data(mask_path)
    mask_dat = mask_dat[::stride, ::stride, ::stride]
    bin_mask = mask_dat > 0

    img_flat = img_dat[:, bin_mask].copy()  # (N, n_vox)
    mu = float(img_flat.mean())
    sd = float(img_flat.std()) + 1e-6

    img_flat_normed = (img_flat - mu) / sd
    np.save(cache, img_flat_normed)
    print(f"Saved cache ({img_flat_normed.nbytes / 1e9:.2f} GB): {cache}")

    return img_flat_normed


def load_feat_table(
    tbl_path: [str | Path], 
    prefer_npz: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    tbl_path = Path(tbl_path)

    if prefer_npz and tbl_path.suffix == ".csv":
        npz_path = tbl_path.with_suffix(".npz")
        if npz_path.exists():
            tbl_path = npz_path

    if tbl_path.suffix == ".npz":
        dat = np.load(tbl_path, allow_pickle=True)
        subjs = dat["SID"].astype(str)
        X = dat["X"].astype(np.float32)

        keep = ~np.isnan(np.asarray(X)).any(axis=1)
        subjs = subjs[keep]
        X = np.ascontiguousarray(np.asarray(X)[keep])

    elif tbl_path.suffix in (".csv", ".tsv"):
        df = pd.read_csv(tbl_path, sep="\t" if tbl_path.suffix == ".tsv" else ",")
        df = df.dropna()
        subjs = df.iloc[:, 0].to_numpy(dtype=str)
        X = df.iloc[:, 1:].to_numpy(dtype=np.float32)

    else:
        raise ValueError(f"Unsupported table format: '{tbl_path.suffix}' ({tbl_path})")

    print(f"Loaded: {tbl_path.name}")
    print(f"{len(subjs)} participants, {X.shape[1]} features")

    return subjs, X


def train_eval_model(
    X: np.ndarray, 
    y: np.ndarray, 
    idx_tr: np.ndarray, 
    idx_te: np.ndarray, 
    model_type: str, 
    seed: int = 42, 
    seed_inner: int = 0, 
    n_folds: int = 5, 
    l1_ratios: list[float] = [.1, .5, .7, .9, .95, .99, 1], 
    alphas: list[float] = [1e-1, 1.0, 3.0, 1e1, 3e1, 1e2, 3e2, 1e3, 1e4, 1e5], 
    max_iter: int = 10000, 
    n_jobs: int = -1, 
    verbose: int = 1, 
    model_path_template: [str | Path] = None, 
    model_perf_path: [str | Path] = None, 
    overwrite: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    '''
    Fit a standardized linear regression model (with built-in CV for regularization strength) 
    and return predictions for both the training and the testing set.

    - For training set, predictions are generated over k-folds from models trained on out-of-fold data
    - For testing set, predictions are generated from a single model trained on all training data
    - Each model is a `StandardScaler` -> `*CV` regressor pipeline;
      the scaler is fitted only on the respective training subset
    - Two nested splits are involved:
      the outer `KFold` over `idx_tr` (`n_folds` folds, seeded by `seed`) that yields out-of-fold predictions,
      and the estimator's inner CV that selects `alpha` / `l1_ratio`
      (always 5-fold regardless of `n_folds`, seeded by `seed_inner`)
    - If `model_path_template` is given, fitted pipelines are cached to disk;
      an existing file is loaded instead of refitted unless `overwrite=True`

    Parameters
    ----------
    - X : np.ndarray, shape (n_samples, n_features)
        Feature matrix. Rows must align with `y`.

    - y : np.ndarray, shape (n_samples,)
        Target values (1-D), e.g. chronological age.

    - idx_tr : np.ndarray of int
        Row indices of `X` / `y` forming the training set. 
        Split further into `n_folds` outer folds to produce out-of-fold predictions.

    - idx_te : np.ndarray of int
        Row indices of the held-out testing set. Must not overlap `idx_tr`.
        Indices in neither array are simply left unpredicted.

    - model_type : {"elasticnet", "lasso", "ridge"}
        Regressor to use: `ElasticNetCV`, `LassoCV`, or `RidgeCV`.

    - seed : int, default 42
        Random state of the outer `KFold` over `idx_tr` (shuffled).

    - seed_inner : int, default 0
        Random state of the estimator's inner `KFold` (shuffled),
        which selects `alpha` / `l1_ratio`.
        Independent of `seed`; keep it fixed across outer folds
        so that hyper-parameter selection is comparable between them.

    - n_folds : int, default 5
        Number of outer folds over `idx_tr`.
        Does not affect the inner CV, which is always 5-fold.

    - l1_ratios : list[float], default [.1, .5, .7, .9, .95, .99, 1]
        Candidate L1/L2 mixing ratios. Used by "elasticnet" only.

    - alphas : list[float], default [1e-1, 1.0, 3.0, 1e1, 3e1, 1e2, 3e2, 1e3, 1e4, 1e5]
        Candidate regularization strengths, searched by the estimator's inner CV.

    - max_iter : int, default 10000
        Maximum coordinate-descent iterations. Used by "elasticnet" / "lasso" only.

    - n_jobs : int, default -1
        Parallel jobs for the inner CV. Used by "elasticnet" / "lasso" only.

    - verbose : int, default 1
        Verbosity of the regressor's inner CV. Used by "elasticnet" / "lasso" only
        (`RidgeCV` accepts neither `verbose` nor `n_jobs`).

    - model_path_template : str or os.PathLike, optional
        Path template for caching fitted pipelines via `joblib.dump` / `joblib.load`;
        must contain a single "{}" placeholder and end with ".joblib".
        Filled with "fold-0" ... "fold-{n_folds-1}" for the outer-fold models
        and "test" for the model trained on all of `idx_tr`.
        Parent directory is created if needed.
        A path that already exists is loaded rather than refitted, unless `overwrite`.
        If omitted, models are neither loaded nor saved.

    - model_perf_path : str or os.PathLike, optional
        Path of a ".csv" file to write performance metrics in long format,
        with columns `Split`, `N`, `MAE`, `R2` and one row per split:
          - "Val_fold-0" ... "Val_fold-{n_folds-1}" : each outer fold's held-out samples
          - "Val_mean" / "Val_SD"                   : mean and sample SD (ddof=1) across those folds
          - "Val_pooled"                            : all out-of-fold predictions on `idx_tr` scored at once
          - "Test"                                  : predictions on `idx_te`
        `N` is the number of samples scored, left empty for the "Val_mean" / "Val_SD" rows.
        Note that "Val_pooled" is not a per-fold average: its MAE equals "Val_mean" only when
        the folds are of equal size, and its R2 is computed against the mean of all of `y[idx_tr]`.
        Parent directory is created if needed.
        Always (re)written when given, regardless of `overwrite`.
        If omitted, metrics are neither computed nor saved.

    - overwrite : bool, default False
        Refit and re-dump every model even when its `model_path_template` file exists.
        No effect when `model_path_template` is omitted.

    Returns
    -------
    - y_pred : np.ndarray, shape (n_samples,), dtype float32
        Predicted values aligned with `y`. 
        Out-of-fold predictions at `idx_tr`,
        full-training-set-model predictions at `idx_te`, 
        `np.nan` elsewhere.
        
    - fold_n : np.ndarray, shape (n_samples,), dtype int8
        Outer-fold index (0 ... `n_folds`-1) 
        at which each training sample was held out; 
        -1 for testing and unused samples.

    Raises
    ------
    AssertionError
        - `X` / `y` lengths mismatch
        - `y` is not 1-D
        - `idx_tr` / `idx_te` are not integer indices, are empty, are out of bounds, or overlap
        - `model_type` is unknown
        - the output paths violate the format constraints above
    '''

    import os
    import joblib
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
    from sklearn.model_selection import KFold

    def _init_model(*args, **kwargs):
        _kf = KFold(n_splits=5, shuffle=True, random_state=seed_inner)

        if model_type == "elasticnet": 
            return make_pipeline(
                StandardScaler(), 
                ElasticNetCV(
                    l1_ratio=l1_ratios, 
                    alphas=alphas, 
                    cv=_kf, 
                    max_iter=max_iter, 
                    n_jobs=n_jobs, 
                    verbose=verbose
                )
            )
        elif model_type == "lasso": 
            return make_pipeline(
                StandardScaler(), 
                LassoCV(
                    alphas=alphas, 
                    cv=_kf, 
                    max_iter=max_iter, 
                    n_jobs=n_jobs, 
                    verbose=verbose
                )
            )
        elif model_type == "ridge": 
            return make_pipeline(
                StandardScaler(), 
                RidgeCV(
                    alphas=alphas, 
                    cv=_kf
                )
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def _calc_model_perf(y: np.ndarray, y_pred: np.ndarray):
        err = y - y_pred
        mae = float(np.mean(np.abs(err)))
        r2 = 1 - float(np.sum(err ** 2)) / float(np.sum((y - y.mean()) ** 2))
        return {"MAE": mae, "R2": r2}

    ## Check inputs validity --------------------------------------------------------

    model_path_template = None if model_path_template is None else str(model_path_template)
    model_perf_path = None if model_perf_path is None else str(model_perf_path)

    assert X.shape[0] == len(y), f"\nMismatch between length of X ({X.shape[0]}) and y ({len(y)})\n"
    assert y.ndim == 1, f"\ny must be 1-D, got shape {y.shape}\n"
    assert idx_tr.dtype.kind in "iu" and idx_te.dtype.kind in "iu", f"\nidx_tr/idx_te must be integer indices, got {idx_tr.dtype}/{idx_te.dtype}\n"

    overlap = np.intersect1d(idx_tr, idx_te)
    assert overlap.size == 0, f"\n{overlap.size} overlap(s) between training and testing indices\n"
    assert model_type in ["elasticnet", "lasso", "ridge"], f"\nmodel_type '{model_type}' is undefined.\n"

    for nm, idx in [("idx_tr", idx_tr), ("idx_te", idx_te)]:
        assert idx.size > 0, f"\n{nm} is empty\n"
        assert idx.min() >= 0 and idx.max() < len(y), f"\n{nm} out of bounds: [{idx.min()}, {idx.max()}] vs len(y)={len(y)}\n"

    if model_path_template:
        assert "{}" in model_path_template, "\n'model_path_template' should have '{}', got:\n" + model_path_template
        assert model_path_template.endswith(".joblib"), f"\n'model_path_template' should end with .joblib, got:\n{model_path_template}"
        _dir_1 = os.path.dirname(model_path_template.format("x"))
        if _dir_1: 
            os.makedirs(_dir_1, exist_ok=True)
    else:
        print("\n'model_path_template' is not defined. No trained model will be saved.\n")

    if model_perf_path:
        assert model_perf_path.endswith(".csv"), f"\n'model_perf_path' should end with .csv, got:\n{model_perf_path}"
        _dir_2 = os.path.dirname(model_perf_path)
        if _dir_2: 
            os.makedirs(_dir_2, exist_ok=True)
    else:
        print("\n'model_perf_path' is not defined. Model performances will not be calculated and saved.\n")

    ## ------------------------------------------------------------------------------

    y_pred = np.full(len(y), np.nan, dtype=np.float32)
    fold_n = np.full(len(y), -1, dtype=np.int8)

    ## For training set
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (tr, va) in enumerate(kf.split(idx_tr)):
        m_cv = None

        if model_path_template:
            m_cv_fp = model_path_template.format(f"fold-{k}")
            if os.path.isfile(m_cv_fp) and not overwrite:
                m_cv = joblib.load(m_cv_fp)
                print(f"Loaded pre-trained: {m_cv_fp}")
                
        if m_cv is None:
            m_cv = _init_model(**locals())
            m_cv.fit(X[idx_tr[tr]], y[idx_tr[tr]])

            if model_path_template:
                joblib.dump(m_cv, m_cv_fp)
                print(f"\nSaved: {m_cv_fp}\n")

        y_pred[idx_tr[va]] = m_cv.predict(X[idx_tr[va]])
        fold_n[idx_tr[va]] = k

    ## For testing set
    m_te = None

    if model_path_template:
        m_te_fp = model_path_template.format("test")
        if os.path.isfile(m_te_fp) and not overwrite:
            m_te = joblib.load(m_te_fp)
            print(f"Loaded pre-trained: {m_te_fp}")

    if m_te is None:
        m_te = _init_model(**locals())
        m_te.fit(X[idx_tr], y[idx_tr])

        if model_path_template:
            joblib.dump(m_te, m_te_fp)
            print(f"\nSaved: {m_te_fp}\n")

    y_pred[idx_te] = m_te.predict(X[idx_te])

    ## Model performance scores
    if model_perf_path:

        perfs = []
        for k in range(n_folds):
            idx_va = idx_tr[fold_n[idx_tr] == k]
            perf_va = _calc_model_perf(y[idx_va], y_pred[idx_va])
            perfs.append({
                "Split": f"Val_fold-{k}",
                "N"    : len(idx_va),
                "MAE"  : perf_va["MAE"],
                "R2"   : perf_va["R2"]
            })

        fold_mae = np.array([ p["MAE"] for p in perfs ])
        fold_r2  = np.array([ p["R2"] for p in perfs ])

        perf_tr = _calc_model_perf(y[idx_tr], y_pred[idx_tr])
        perf_te = _calc_model_perf(y[idx_te], y_pred[idx_te])
        perfs += [
            {"Split": "Val_mean"  , "N": None       , "MAE": float(fold_mae.mean())     , "R2": float(fold_r2.mean())},
            {"Split": "Val_SD"    , "N": None       , "MAE": float(fold_mae.std(ddof=1)), "R2": float(fold_r2.std(ddof=1))},
            {"Split": "Val_pooled", "N": len(idx_tr), "MAE": perf_tr["MAE"]             , "R2": perf_tr["R2"]},
            {"Split": "Test"      , "N": len(idx_te), "MAE": perf_te["MAE"]             , "R2": perf_te["R2"]}
        ]

        perfs = pd.DataFrame(perfs)
        perfs["N"] = perfs["N"].astype("Int64")  # nullable int
        perfs.to_csv(model_perf_path, index=False)
        print(f"\nSaved: {model_perf_path}\n")

    return y_pred, fold_n


